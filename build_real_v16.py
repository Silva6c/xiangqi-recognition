# -*- coding: utf-8 -*-
"""v16/v17 真实集拼接（v16~v17 计划 C1/D1）。

两个模式（--mode）：
- v16：原生屏幕帧 ×4 过采样 + 移植集（v14 配方原样，帧库更新到含残局/对弈新帧）。
       过采样修复移植样本对原生曝光的稀释（v12 教训），产物 real_frames_v16.npz。
- mix：屏幕原生 + 实拍伪标签 + 照片贴图合成 三路 1:1:1 混合（v17 实拍域注入用），
       产物 real_photo_mix.npz。

数据契约：imgs [N,315,280,3] u8 RGB + labels [N,90] int64（IGNORE=-100）。
用法：python build_real_v16.py [--mode v16|mix]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCREEN = HERE / "data/real_frames_labeled.npz"
TRANSPLANTED = HERE / "data/real_frames_transplanted.npz"
PHOTO_TRAIN = HERE / "data/real_photo_train.npz"
COMPOSITE = HERE / "data/photo_composite.npz"
OUT_V16 = HERE / "data/real_frames_v16.npz"
OUT_MIX = HERE / "data/real_photo_mix.npz"


def load(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        print(f"  [缺] {path.name} 不存在，跳过该路")
        return None
    z = np.load(path, allow_pickle=False)
    print(f"  {path.name}: {len(z['imgs'])} 样本")
    return z["imgs"], z["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v16", "mix", "v18"], default="v16")
    ap.add_argument("--oversample", type=int, default=4, help="v16 原生过采样倍数")
    args = ap.parse_args()

    parts: list[tuple[np.ndarray, np.ndarray]] = []
    if args.mode == "v16":
        print("v16 拼接（v14 配方）：原生×4 全量 + 移植全量（不截断）")
        screen = load(SCREEN)
        if screen is None:
            raise SystemExit("先跑 annotate_real.py")
        imgs, labs = screen
        parts.append((np.concatenate([imgs] * args.oversample),
                      np.concatenate([labs] * args.oversample)))
        tr = load(TRANSPLANTED)
        if tr:
            parts.append(tr)
        out = OUT_V16
        n_min = None                                # v16 模式保留各路全量
    elif args.mode == "v18":
        # v18：屏域（原生×4 + 移植扩充）+ 实拍扩容（主集×2 + 回收难例×2）+ 贴图×2。
        # 屏域 : 实拍系 ≈ 2:1，实拍系从 v17 的 1050 扩到 ~3900；recover0.55 源图
        # 已被 recover45 全覆盖，不重复收录。
        print("v18 拼接：原生×4 + 移植全量 + 实拍主集×2 + 回收45×2 + 贴图v18×2")
        screen = load(SCREEN)
        if screen is None:
            raise SystemExit("先跑 annotate_real.py")
        imgs, labs = screen
        parts.append((np.concatenate([imgs] * args.oversample),
                      np.concatenate([labs] * args.oversample)))
        for extra in (TRANSPLANTED,
                      PHOTO_TRAIN, PHOTO_TRAIN,
                      HERE / "data/real_photo_recover45.npz",
                      HERE / "data/real_photo_recover45.npz",
                      HERE / "data/photo_composite_v18.npz",
                      HERE / "data/photo_composite_v18.npz"):
            got = load(extra)
            if got:
                parts.append(got)
        out = HERE / "data/real_frames_v18.npz"
        n_min = None
    else:
        print("mix 拼接（v17）：屏幕原生 + 实拍伪标签 + 照片贴图合成（1:1:1）")
        for p in (SCREEN, PHOTO_TRAIN, COMPOSITE):
            got = load(p)
            if got:
                parts.append(got)
        out = OUT_MIX
        n_min = min(len(i) for i, _ in parts)       # 三路均衡防单路淹没

    if not parts:
        raise SystemExit("没有可用的数据路")

    if n_min is not None:
        imgs = np.concatenate([i[:n_min] for i, _ in parts])
        labs = np.concatenate([l[:n_min] for _, l in parts])
    else:
        imgs = np.concatenate([i for i, _ in parts])
        labs = np.concatenate([l for _, l in parts])
    order = np.random.RandomState(20261003).permutation(len(imgs))
    imgs, labs = imgs[order], labs[order]

    np.savez_compressed(out, imgs=imgs.astype(np.uint8), labels=labs.astype(np.int64))
    size_mb = out.stat().st_size / 1e6
    src = f"（{len(parts)} 路" + (f" × {n_min} 均衡）" if n_min else "全量）")
    print(f"落盘 {out}: {len(imgs)} 样本{src}，{size_mb:.0f}MB")


if __name__ == "__main__":
    main()
