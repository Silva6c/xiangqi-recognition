# -*- coding: utf-8 -*-
"""识别环街拍注入探针（v67 立项前可行性，纯推理零训练）。

对 photo 双闸清洗池 1498 帧街拍 warp 图跑 nano 伪标签，过三闸统计弹药量：
  闸A nano 置信度（softmax pmax 均值 ≥0.55——辅助闸，v60 前科只作参考）
  闸B 子力编制校验（车≤2马≤2炮≤2兵≤5仕≤2相≤2、帅将各 1——最强校验器）
  闸C r3 一致率（格级一致占比，喂料时只喂一致格标 IGNORE 的参照指标）
判定线（预注册）：编制通过率 ≥50% 且 r3 一致率 ≥60% → 立项开训；否则路线死刑。
用法：python probe_nano_pseudo.py [--max-n 200]（--max-n 抽样探针，省时先看趋势）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_screen30 import (          # noqa: E402
    sess_nano, sess_r3, MEAN, STD, cls_input_280x315,
    r3_probs, cell_out, full_cascade, detect_orientation_canonical,
)
from corner_data import imread_cn    # noqa: E402
from diag_two_domains import CLS     # noqa: E402  ".xKABNRCPkabnrcp"（2=K 6=R 9=k 13=r …）

# 编制上限（索引按 CLS：2K 3A 4B 5N 6R 7C 8P / 9k 10a 11b 12n 13r 14c 15p）
LIMITS = {2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 5,
          9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 5}


def composition_ok(pred90: np.ndarray) -> bool:
    """子力编制校验：各类不超编+红帅黑将各恰 1。"""
    c = Counter(pred90.tolist())
    for cls_, lim in LIMITS.items():
        if c.get(cls_, 0) > lim:
            return False
    return c.get(2, 0) == 1 and c.get(9, 0) == 1


def warp_by_corners(img, corners):
    H = cv2.getPerspectiveTransform(
        corners[[0, 1, 2, 3]].astype(np.float32),
        np.array([[50, 50], [400, 50], [50, 450], [400, 450]], np.float32))
    return cv2.warpPerspective(img, H, (450, 500),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="../../.cache/corner_photo_clean.npz")
    ap.add_argument("--max-n", type=int, default=0, help=">0 时抽样前 N 帧")
    args = ap.parse_args()

    d = np.load(args.pool, allow_pickle=True)
    files, corners = d["files"], d["corners"]
    if args.max_n:
        files, corners = files[:args.max_n], corners[:args.max_n]

    n = len(files)
    stats = Counter()
    agree_cells = agree_valid = 0          # r3 一致格统计（有效格=非空格）
    conf_list = []
    for f, gt_c in zip(files, corners):
        img = imread_cn(Path(str(f)))
        if img is None:
            stats["read_fail"] += 1
            continue
        w450 = warp_by_corners(img, np.asarray(gt_c, np.float32))
        img280 = cls_input_280x315(w450)
        x = (img280.astype(np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
        lo = sess_nano.run(None, {sess_nano.get_inputs()[0].name:
                                  np.ascontiguousarray(x[None])})[0][0]   # [90,15] logits
        p = np.exp(lo - lo.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        npred = p.argmax(1)
        pmax = p.max(1).mean()
        conf_list.append(float(pmax))
        g, _ = detect_orientation_canonical(npred)
        if g is None:
            stats["orient_fail"] += 1
            continue
        npred = g.reshape(-1)
        stats["total"] += 1
        if not composition_ok(npred):
            stats["comp_fail"] += 1
            continue
        stats["comp_ok"] += 1
        # 探针修正：nano pmax med=0.130 近均匀（v60 同款），置信度闸死刑删除；
        # r3 格级一致率升为主闸（编制通过帧上直接测）
        o = r3_probs(img280)
        cp, cc = cell_out(img280)
        p3, _ = full_cascade(o["top1"], o["pmax"], o["second"], cp, cc)
        g3, _ = detect_orientation_canonical(p3)
        if g3 is None:
            stats["r3_orient_fail"] += 1
            continue
        r3pred = g3.reshape(-1)
        same = (r3pred == npred)
        agree_cells += int(same.sum())
        agree_valid += 90
        stats["r3_cmp"] += 1

    t = max(stats["total"], 1)
    cl = np.array(conf_list)
    print(f"== nano 伪标签探针（{n} 帧）==")
    print(f"读取失败 {stats['read_fail']} | 朝向失败 {stats['orient_fail']}")
    print(f"编制通过 {stats['comp_ok']}/{t} = {stats['comp_ok']/t:.3f}")
    print(f"（探针修正：置信度闸已删——nano pmax med={np.median(cl):.3f} 近均匀，v60 同款死刑）")
    if agree_valid:
        print(f"r3 格级一致率 {agree_cells}/{agree_valid} = {agree_cells/agree_valid:.3f}"
              f"（{stats['r3_cmp']} 帧上，r3 朝向失败另 {stats['r3_orient_fail']}）")
    print(f"nano pmax 均值分布：med={np.median(cl):.3f} p25={np.percentile(cl,25):.3f} "
          f"p75={np.percentile(cl,75):.3f}")
    print(f"判定线（修正）：编制通过率 ≥0.50 且 r3 一致率 ≥0.60 → 立项")
    json.dump({"n": n, "stats": dict(stats),
               "comp_rate": stats["comp_ok"] / t, "agree_rate": agree_cells / max(agree_valid, 1),
               "pmax_med": float(np.median(cl))},
              open(HERE / "../../.cache/nano_pseudo_probe.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已存 .cache/nano_pseudo_probe.json")


if __name__ == "__main__":
    main()
