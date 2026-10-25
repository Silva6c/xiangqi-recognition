# -*- coding: utf-8 -*-
"""统计渲染分布的格-类频率表（v41 格条件类平衡损失的数据依赖）。

产出的 [90,16] 每格归一化频率表供 train.py --cell-balance 使用。
**renderer 增广改动后必须重跑本脚本**——频率表必须与训练时的真实渲染分布一致，
否则类平衡权重就是错的。

用法：
  python stat_cell_freq.py --n 6000 --out data/cell_class_freq.npz
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import renderer as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="采样帧数（越大越稳，6000 约 8 分钟）")
    ap.add_argument("--out", default=str(Path(__file__).parent / "data/cell_class_freq.npz"))
    ap.add_argument("--seed0", type=int, default=2_000_000_000, help="起始种子（与训练流独立）")
    args = ap.parse_args()

    fonts = R.load_font_pool()
    freq = np.zeros((R.ROWS * R.COLS, len(R.CLASS_CHARS)), np.float64)
    for i in range(args.n):
        rng = random.Random(args.seed0 + i)
        s = R.make_sample(rng, fonts)
        lab = s["labels"]
        valid = lab != R.IGNORE_LABEL
        freq[np.arange(len(lab))[valid], lab[valid]] += 1
    freq_n = freq / np.maximum(freq.sum(axis=1, keepdims=True), 1e-9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, freq=freq_n, n_samples=args.n, seed0=args.seed0)
    # 报告几个战场格
    for cell, tag in ((88, "(9,7)"), (82, "(9,1)"), (89, "(9,8)"), (7, "(0,7)")):
        top = ", ".join(f"{R.CLASS_CHARS[k]} {freq_n[cell, k]*100:.1f}%"
                        for k in np.argsort(-freq_n[cell])[:4])
        print(f"  {tag}: {top}")
    print(f"频率表 → {out}（{args.n} 帧）")


if __name__ == "__main__":
    main()
