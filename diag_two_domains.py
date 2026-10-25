# -*- coding: utf-8 -*-
"""bench 三域诊断：v9 学生在（A 原始域 / B 修正域 / R 回退域）上的错格对比。

目的：bench 实际输入（域 A）上每帧固定错 2-3 个边缘格，而 v9 训练增广的
bench 型失真分支只有 ~0.2% 样本命中"精确 bench 模式"（30% 密度 × 方向 25% ×
幅度命中率），疑似覆盖不足。本脚本用三个域的错格数反推失真精确方向与幅度：
- 若 R 域（把 bench 失真向 canonical 回拉）错格显著减少 → bench 失真超出学生
  增广覆盖带，且幅度 = SCALE_CORRECTIONS 的逆；
- 若 B 域（失真加重）错格更多 → 同上佐证。

用法：python diag_two_domains.py [--model runs/v9/gridhead_fp32.onnx]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from pose_corner import warp_frame_to_warp450  # noqa: E402
import renderer as R  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
CLS = ".xKABNRCPkabnrcp"


def shrink_corners(corners: np.ndarray, scale, delta) -> np.ndarray:
    """与 annotate_real.shrink_corners 同义：角点绕中心缩放 + 平移（原图域）"""
    c = corners.mean(axis=0)
    return (corners - c) * np.array(scale) + c + np.array(delta)


def fen_to_labels(fen: str) -> np.ndarray:
    """标准 FEN → 90 格标签（行 0=黑方底线，类序 .xKABNRCPkabnrcp）"""
    lab = np.full(90, R.IGNORE_LABEL, np.int64)
    rows = fen.split()[0].split("/")
    for i, row in enumerate(rows):
        j = 0
        for ch in row:
            if ch.isdigit():
                j += int(ch)
            else:
                lab[i * 9 + j] = CLS.index(ch)
                j += 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v9/gridhead_fp32.onnx")
    args = ap.parse_args()

    zb = np.load(Path(__file__).parent / "data/eval_bench.npz", allow_pickle=False)
    meta = json.loads(str(zb["meta"]))
    case_ids, quads = meta["case_ids"], meta["quads"]
    cases = json.load(open(REPO / "testdata/recog-bench/cases.json", encoding="utf-8"))["cases"]
    fen_by_id = {c["id"]: c["fen"] for c in cases}

    sess = ort.InferenceSession(str(Path(__file__).parent / args.model),
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    DOMAINS = {
        "A 原始域": ((1.0, 1.0), (0.0, 0.0)),
        "B 修正域1.029": ((1.029, 1.005), (23.8, 9.5)),
        "R 回退域": ((1 / 1.029, 1 / 1.005), (-23.8, -9.5)),
    }
    err_stat = {k: [] for k in DOMAINS}

    for cid, q in zip(case_ids, quads):
        png = REPO / ".shots/bench" / f"case{cid:02d}.png"
        img = cv2.imread(str(png))
        if img is None:
            print(f"case{cid:02d}: 底片缺失，跳过")
            continue
        corners = np.array(q, np.float64)
        lab = fen_to_labels(fen_by_id[cid])
        line = [f"case{cid:02d}"]
        for dname, (sc, dl) in DOMAINS.items():
            cc = shrink_corners(corners, sc, dl)
            w450 = warp_frame_to_warp450(img, cc.astype(np.float32))
            cls_img = R.warp450_to_cls(w450)
            x = cls_img.astype(np.float32).transpose(2, 0, 1) / 255.0
            x = (x - IM_MEAN) / IM_STD
            pred = sess.run(None, {iname: x[None].astype(np.float32)})[0].reshape(90, 16).argmax(1)
            errs = [(i // 9, i % 9) for i in range(90)
                    if lab[i] != R.IGNORE_LABEL and pred[i] != lab[i]]
            err_stat[dname].append(len(errs))
            line.append(f"{dname.split()[0]}:{len(errs)}")
        print("  ".join(line))

    print()
    for dname, e in err_stat.items():
        e = np.array(e)
        print(f"{dname}: 总错格 {e.sum()}  均值 {e.mean():.1f}/帧  全对帧 {(e == 0).sum()}/10")


if __name__ == "__main__":
    main()
