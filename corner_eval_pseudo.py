# -*- coding: utf-8 -*-
"""第 1 波验收：真实域 heldout 一致率（自研角点 vs RTMPose 伪标签，≤15px 算一致）。

判据（预注册）：photo 域一致率 ≥98%（RTMPose 在真实域可信——26 帧级联 1 错实证）；
screen 域一并报告（参考线，无硬线——RTMPose 竖屏半盘缺陷域）。
用法：python corner_eval_pseudo.py --corner-pt runs/corner_v2/best.pt
      python corner_eval_pseudo.py --onnx runs/corner_v2/corner_simcc_fp32.onnx
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
from corner_infer import CornerInfer
from corner_data import imread_cn

THR_PX = 15.0


def corner_dist(det, gt):
    """模型序 [TL,TR,BL,BR] 逐角 L2 距离均值（原图坐标）。"""
    return float(np.linalg.norm(det - gt, axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner-pt", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--pseudo", default=str(HERE / "data/pools/corner_pseudo.npz"))
    args = ap.parse_args()
    inf = CornerInfer(torch_weights=args.corner_pt, onnx_path=args.onnx)

    d = np.load(args.pseudo, allow_pickle=True)
    keep = d["split"] == "holdout"
    files, gts, doms = d["files"][keep], d["corners"][keep], d["domain"][keep]
    stats = {}
    for f, gt, dom in zip(files, gts, doms):
        img = imread_cn(HERE / str(f))
        det = inf.detect_with_check(img)
        if det is None:
            ok, dist = False, 9999.0
        else:
            dist = corner_dist(det, gt)
            ok = dist <= THR_PX
        s = stats.setdefault(str(dom), {"n": 0, "ok": 0, "dists": []})
        s["n"] += 1
        s["ok"] += int(ok)
        s["dists"].append(dist)
    report = {}
    for dom, s in stats.items():
        ds = np.array(s["dists"])
        report[dom] = {"n": s["n"], "agree": s["ok"], "rate": s["ok"] / s["n"],
                       "dist_med": float(np.median(ds)), "dist_p90": float(np.percentile(ds, 90))}
        print(f"[{dom}] 一致率 {s['ok']}/{s['n']} = {s['ok']/s['n']:.4f} "
              f"中位距离 {np.median(ds):.1f}px p90 {np.percentile(ds, 90):.1f}px")
    (HERE / "data/reports").mkdir(parents=True, exist_ok=True)
    (HERE / "data/reports/corner_pseudo_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已存 .cache/corner_pseudo_eval.json")


if __name__ == "__main__":
    main()
