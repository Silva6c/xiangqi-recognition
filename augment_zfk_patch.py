# -*- coding: utf-8 -*-
"""v46 zfk 三材质原图裁片管线（docs/v46方案-实物最后一格与bench攻坚-20261018.md §二A）。

背景：zfk 检测集（888 图 28302 框，亚克力/不锈钢/木质三材质——26 帧实物域没有的
成像条件）是 YOLO txt 标注，augment_vn_raw_patch.py 的 COCO 直读不适用。本脚本
复用其 QC 语义（开局标准位换子自证：nano batch=1 双门 argmax==类 + conf≥0.65，
坏片在库端全拦），仅替换裁片来源：

 1. 遍历 labels/*.txt（`class_id cx cy w h` 归一化）→ 像素框 → 外扩 25% → resize 38×38；
 2. 类别经 meta.json 的 class_map_16（14→16 类字符，import_zfk.py 核好的映射）；
 3. 图片级 md5 去重（v35 实测 ~158 张同 md5 近重复帧，固定机位抽帧）；
 4. K/k（帅将框）源头不入队——防双帅 BAN 同主库规则（VN 是入队后不消费，这里
    源头跳过省推理，统计单列）；
 5. QC 主循环与 VN 同构（渲染底帧 ≥24 占子格 → 占子格按类贴片 → nano 验全格）。

产出 data/zfk_raw_patch_lib.npz（cls_idx→[N,38,38,3] u8），供
augment_roboflow_patch.py --extra-lib 合并（v46 起支持逗号分隔多库）。

用法：
  python augment_zfk_patch.py --limit-frames 40   # 冒烟
  python augment_zfk_patch.py                     # 全量 28302 框
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import augment_real_transplant as AT
import renderer as R
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
ROOT = HERE.parent.parent.parent / "data/external/chess_material_zfk"
OUT_NPZ = HERE / "data/zfk_raw_patch_lib.npz"
BAN = ("K", "k")

CLS_CHARS = ".xKABNRCPkabnrcp"
IDX = {ch: i for i, ch in enumerate(CLS_CHARS)}


def crop_boxes() -> tuple[dict[int, deque], Counter]:
    """遍历 zfk labels/*.txt 裁全部候选框 → 按类队列。返回 (queues, stat)。"""
    meta = json.load(open(ROOT / "meta.json", encoding="utf-8"))
    cls_map = {int(k): v for k, v in meta["class_map_16"].items()}
    queues: dict[int, deque] = defaultdict(deque)
    stat = Counter()
    seen_md5: set[str] = set()
    for tp in sorted(ROOT.glob("labels/*.txt")):
        ip = ROOT / "images" / (tp.stem + ".jpg")
        img = cv2.imread(str(ip))
        if img is None:
            stat["读取失败"] += 1
            continue
        # 图片级 md5 去重：固定机位抽帧的整帧重复在源头拦掉
        md5 = hashlib.md5(ip.read_bytes()).hexdigest()
        if md5 in seen_md5:
            stat["md5重复图"] += 1
            continue
        seen_md5.add(md5)
        h, w = img.shape[:2]
        for line in tp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 4 and len(parts) != 5:
                continue
            cid = int(parts[0])
            ch = cls_map.get(cid)
            if ch is None:
                stat["类外"] += 1
                continue
            if ch in BAN:
                stat["KkBAN"] += 1
                continue
            # YOLO 归一化中心框 → 像素 xywh
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            bw, bh = bw * w, bh * h
            if bw < 8 or bh < 8:
                stat["框过小"] += 1
                continue
            # 外扩 25% 保棋子完整（含边缘光照），clip 到图边界
            ex, ey = bw * 0.25, bh * 0.25
            x0, y0 = max(0, int(cx * w - bw / 2 - ex)), max(0, int(cy * h - bh / 2 - ey))
            x1, y1 = min(w, int(cx * w + bw / 2 + ex)), min(h, int(cy * h + bh / 2 + ey))
            if x1 - x0 < 8 or y1 - y0 < 8:
                stat["裁片越界"] += 1
                continue
            patch = cv2.resize(img[y0:y1, x0:x1], (38, 38))
            queues[IDX[ch]].append(patch)
            stat["候选"] += 1
            stat[f"候选_{tp.stem.split('_')[0]}"] += 1
    return queues, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_NPZ))
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="QC 底帧数上限（0=按需全量；冒烟用小值）")
    ap.add_argument("--min-conf", type=float, default=0.65)
    args = ap.parse_args()
    t0 = time.time()

    queues, stat = crop_boxes()
    total = sum(len(q) for q in queues.values())
    print(f"裁片候选：{total} 枚（{dict(stat)}）", flush=True)
    print(f"按类：{ {CLS_CHARS[i]: len(q) for i, q in sorted(queues.items())} }", flush=True)

    sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    lib: dict[int, list[np.ndarray]] = defaultdict(list)
    rng = random.Random(2027)
    n_qc = 0
    n_test = 0
    seed = 95_000

    while total := sum(len(q) for q in queues.values()):
        if args.limit_frames and n_qc >= args.limit_frames:
            break
        # 渲一帧底帧（要求 ≥24 占子格的整盘；opening/midgame 通常满足）
        s = None
        for _ in range(8):
            cand = R.make_sample(random.Random(seed), T.font_pools()[0])
            seed += 1
            if int((cand["labels"] > 0).sum()) >= 24:
                s = cand
                break
        n_qc += 1
        if s is None:
            continue
        bgr = cv2.cvtColor(s["cls_img"], cv2.COLOR_RGB2BGR)
        placed = []                                     # (cell, cls_idx, patch)
        cells = list(range(90))
        rng.shuffle(cells)
        for c in cells:
            v = int(s["labels"][c])
            if v <= 0 or v == IDX["."] or not queues.get(v):
                continue
            patch = queues[v].popleft()
            cx, cy = AT.cls_cell_center(c // 9, c % 9)
            AT.paste_piece(bgr, patch, cx, cy)
            placed.append((c, v, patch))
        if not placed:
            continue
        # nano 单帧推理（batch=1——BN 批污染坑）
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        x = np.ascontiguousarray(((x - T.IM_MEAN_NP) / T.IM_STD_NP)[None])
        lo = sess.run(None, {iname: x})[0][0]
        for c, ci, patch in placed:
            n_test += 1
            row = lo[c].astype(np.float64)
            row = row / max(1e-9, row.sum())
            if int(lo[c].argmax()) == ci and float(row[ci]) >= args.min_conf:
                lib[ci].append(patch)
        if n_qc % 100 == 0:
            print(f"  QC {n_qc} 帧，入库 {sum(len(v) for v in lib.values())}/{n_test}，"
                  f"队列余 {total}", flush=True)

    dist = {CLS_CHARS[i]: len(v) for i, v in sorted(lib.items())}
    n_lib = sum(len(v) for v in lib.values())
    print(f"QC 完成：{n_qc} 帧底帧，入库 {n_lib}/{n_test}（{n_lib/max(1,n_test):.0%}），"
          f"覆盖 {len(lib)} 类 {dist}，{time.time()-t0:.0f}s", flush=True)
    if not lib:
        raise SystemExit("zfk 贴片库为空：检查映射/QC 阈值")
    out = {f"cls{i}": np.stack(v) for i, v in lib.items()}
    out["meta"] = np.array([json.dumps({
        "n_lib": n_lib, "n_test": n_test, "n_qc": n_qc,
        "min_conf": args.min_conf, "dist": dist, "stat": dict(stat),
    }, ensure_ascii=False)])
    np.savez_compressed(args.out, **out)
    print(f"落盘 {args.out}（{Path(args.out).stat().st_size/1e6:.1f}MB）")


if __name__ == "__main__":
    main()
