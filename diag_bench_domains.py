# -*- coding: utf-8 -*-
"""bench 双域诊断：v3 在 原生域(1.0) vs 校正域(0.92) 的错格对比；nano_v3 错格定位。

回答一个问题：若部署侧给 RTMPose 角点加 0.92 中心收缩校正，v3 的 bench 表现能否显著提升？
（提升 → 阶段 3 在端上 warp 前加同款校正即通向 100%；不提升 → 继续堆真实帧/渲染迭代。）
用法：python diag_bench_domains.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import pose_corner as PC
import renderer as R
from annotate_real import IM_MEAN, IM_STD, NANO_V3, shrink_corners

HERE = Path(__file__).resolve().parent
REPO = HERE  # 平铺布局：脚本即仓根
V3 = HERE / "runs/v3/gridhead_fp32.onnx"
CLS = R.CLASS_CHARS


def infer(m: ort.InferenceSession, img280: np.ndarray) -> np.ndarray:
    x = img280.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    x = np.ascontiguousarray((x - IM_MEAN) / IM_STD)
    return m.run(None, {m.get_inputs()[0].name: x})[0].reshape(90, 16)


def main():
    cases = json.loads((REPO / "testdata/recog-bench/cases.json").read_text(encoding="utf-8"))["cases"]
    det = PC.PoseCornerDetector()
    nano = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    v3 = ort.InferenceSession(str(V3), providers=["CPUExecutionProvider"])

    per_domain: dict[str, list[int]] = {"1.0": [], "0.92": []}
    for case in cases:
        cid = case["id"]
        png = REPO / f".shots/bench/case{cid:02d}.png"
        if not png.exists():
            continue
        frame = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        corners = det.detect_with_check(frame)
        if corners is None:
            continue
        fen_rows = case["fen"].split()[0].split("/")
        lab = np.full(90, R.CLASS_INDEX["."], np.int64)
        for r, row in enumerate(fen_rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    lab[r * 9 + c] = R.CLASS_INDEX[ch]
                    c += 1
        for sc, key in [(1.0, "1.0"), (0.92, "0.92")]:
            w450 = PC.warp_frame_to_warp450(frame, shrink_corners(corners, sc))
            img = R.warp450_to_cls(w450)
            if sc == 1.0:  # nano 只跑原生域（它的训练域）
                p_nano = infer(nano, img).argmax(1)
                errs = [(j // 9, j % 9, CLS[lab[j]], CLS[p_nano[j]]) for j in range(90) if p_nano[j] != lab[j]]
                if errs:
                    print(f"case{cid} nano错格(行,列,真,测): {errs[:8]}")
            p = infer(v3, img).argmax(1)
            n_bad = int((p != lab).sum())
            per_domain[key].append(n_bad)
            if sc == 0.92:
                verr = [(j // 9, j % 9, CLS[lab[j]], CLS[p[j]]) for j in range(90) if p[j] != lab[j]]
                if verr:
                    print(f"case{cid} v3@0.92错格: {verr[:10]}")
    tot = {k: sum(v) for k, v in per_domain.items()}
    print(f"\nv3 每帧错格: 原域={per_domain['1.0']}")
    print(f"v3 每帧错格: 校正域={per_domain['0.92']}")
    print(f"总错格: 原域 {tot['1.0']} vs 校正域 {tot['0.92']}")


if __name__ == "__main__":
    main()
