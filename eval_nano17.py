# -*- coding: utf-8 -*-
"""nano 上游 sota17 对照评测（v65）：origin 17 张真实照片/软件棋盘图。

三口径：
  ① 自研端到端：v4 角点→warp→r3 级联→朝向 canonical→逐子比对真值
  ② 识别环 nano 底：同 warp 喂 nano（与自研同定位，公平对照识别环）
  ③ 汇总 full-err-k（错 ≤k 子局占比）对齐上游 sota_comparison.md 口径
     （cchess_reg 13/17 零错、最多 3 子；天天象棋 9/17；棋路 10/17）
真值：.cache/nano17_truth.json（10 行×9 列 kabnrcp 记法，黑上红下）。
用法：python eval_nano17.py [--corner-pt runs/corner_v4/best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_screen30 import (  # noqa: E402  复用其模型加载与管线（模块级加载 r3/nano/cell_cls）
    cls_input_280x315, r3_probs, nano_pred, cell_out,
    full_cascade, detect_orientation_canonical,
)
import eval_screen30 as _ES                   # noqa: E402  sess_nano 等模块级状态经此访问
from corner_infer import CornerInfer          # noqa: E402
from diag_two_domains import CLS              # noqa: E402  ".xKABNRCPkabnrcp"

ORIGIN = HERE / "data/external/chinese-chess-recognition/docs/sota/origin"
TRUTH = HERE / "data/nano17_truth.json"
OUT = HERE / "data/reports/nano17_eval.json"


def truth_labels(board_rows):
    """真值 10 行×9 字符 → 90 标签 + 不定格掩码（x/异常字符跳过不计错）。"""
    lab, skip = np.zeros(90, np.int64), np.zeros(90, bool)
    for i, row in enumerate(board_rows):
        for j, ch in enumerate(row[:9]):
            if ch in CLS and ch != "x":
                lab[i * 9 + j] = CLS.index(ch)
            else:
                skip[i * 9 + j] = True          # x=上游识别不到的占位，评测跳过
    return lab, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner-pt", default="runs/corner_v4/best.pt")
    args, _ = ap.parse_known_args()   # 容忍 --r3 等透传参数（eval_screen30 模块级消费）
    inf = CornerInfer(torch_weights=Path(args.corner_pt))

    truth = json.loads(Path(TRUTH).read_text(encoding="utf-8"))
    rows_out = []
    stats = {"e2e": [], "nano": [], "locate_fail": []}
    for k in sorted(truth, key=int):
        img_path = next((ORIGIN / f"{k}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")
                         if (ORIGIN / f"{k}{ext}").exists()), None)
        if img_path is None:
            print(f"[{k}] 缺 origin 图，跳过"); continue
        img = cv2.imread(str(img_path))
        if img is None:
            from corner_data import imread_cn
            img = imread_cn(img_path)
        lab, skip = truth_labels(truth[k]["board"])

        det = inf.detect_with_check(img)
        if det is None:
            stats["locate_fail"].append(k)
            e2e_err = nano_err = 90
            rows_out.append({"k": k, "locate": "FAIL", "e2e_err": e2e_err, "nano_err": nano_err,
                             "conf": truth[k].get("conf", "")})
            print(f"[{k}] 定位 FAIL（按满错 90 计）")
            continue
        w450 = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(
                det[[0, 1, 2, 3]].astype(np.float32),
                np.array([[50, 50], [400, 50], [50, 450], [400, 450]], np.float32)),
            (450, 500), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img280 = cls_input_280x315(w450)

        # ① 自研端到端：r3 → 级联 → 朝向 canonical
        o = r3_probs(img280)
        cp, cc = cell_out(img280)
        p3, _ = full_cascade(o["top1"], o["pmax"], o["second"], cp, cc)
        g, orient = detect_orientation_canonical(p3)
        if g is None:
            e2e_err = 90
        else:
            pred = g.reshape(-1)
            e2e_err = int((pred[~skip] != lab[~skip]).sum())
        # ② nano 底（同 warp）——可缺省（国产化后 assets 已删 nano onnx）
        if _ES.sess_nano is not None:
            npred, norient = detect_orientation_canonical(nano_pred(img280))
            nano_err = 90 if npred is None else int((npred.reshape(-1)[~skip] != lab[~skip]).sum())
        else:
            nano_err = -1

        stats["e2e"].append(e2e_err); stats["nano"].append(nano_err)
        rows_out.append({"k": k, "locate": "ok", "orient": orient or "",
                         "e2e_err": e2e_err, "nano_err": nano_err,
                         "conf": truth[k].get("conf", "")})
        print(f"[{k}] {Path(img_path).name} 定位ok 朝向{orient} 自研错{e2e_err} "
              f"nano错{'缺省' if nano_err < 0 else nano_err} "
              f"真值置信{truth[k].get('conf','')}")

    n = len(rows_out)
    def fek(errs, k): return sum(1 for e in errs if e <= k)
    e, nn = stats["e2e"], stats["nano"]
    nnv = [x for x in nn if x >= 0]   # nano 缺省帧（-1）不进统计
    print(f"\n== 汇总（{n} 局，定位失败 {len(stats['locate_fail'])}）==")
    print(f"自研端到端 full-err-0/1/3: {fek(e,0)}/{n} {fek(e,1)}/{n} {fek(e,3)}/{n}  "
          f"中位错 {np.median(e):.0f}")
    if nnv:
        print(f"nano 识别环 full-err-0/1/3: {fek(nnv,0)}/{len(nnv)} {fek(nnv,1)}/{len(nnv)} "
              f"{fek(nnv,3)}/{len(nnv)}  中位错 {np.median(nnv):.0f}")
    else:
        print("nano 识别环: 缺省（assets 已删 nano onnx）")
    print("上游对照: cchess_reg 13/17 零错(最多3) | 天天象棋 9/17 | 棋路 10/17")
    json.dump({"rows": rows_out, "summary": {
        "n": n, "locate_fail": stats["locate_fail"],
        "e2e_full_err": [fek(e, 0), fek(e, 1), fek(e, 3)], "nano_full_err": [fek(nn, 0), fek(nn, 1), fek(nn, 3)]}},
        open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已存 {OUT}")


if __name__ == "__main__":
    main()
