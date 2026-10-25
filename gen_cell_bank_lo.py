# -*- coding: utf-8 -*-
"""v2 数据：real 12 池 warp 帧抠格 bank（LO 分布，40px 原始 crop）。

真实域 LO 格是 cell_cls 的大杠杆——v1 只有贴片（38px 特写）缺 warp 帧格分布。
每帧抽 10 格（含空格），40×40×3 u8 存储（训练时上采样 128）。
产出 data/cell_bank_lo.npz（cells[N,40,40,3] + labels[N] + frame_src[N]）。
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
POOLS = ["real_photo_mix_clean_v60", "real_photo_v24_clean_v60", "real_photo_chessai_clean",
         "real_chessai_piece_clean", "real_rf_piece_hi_clean", "bench_hot_transplant",
         "bench_hot_transplant_v3", "empty_paste_v50", "real_photo_v47_clean",
         "real_rf_patch_transplant_v47_shard0", "real_rf_patch_transplant_v47_shard1",
         "real_rf_patch_dense_v58_shard0"]
PER_FRAME = 10
rng = random.Random(310)


def crop_frame(img315: np.ndarray, labels):
    """与 cell_eval.crop_cells 同格心公式，抽 PER_FRAME 格 40px crop。"""
    idxs = [i for i in range(90) if labels[i] != -100]
    rng.shuffle(idxs)
    out, labs = [], []
    for i in idxs[:PER_FRAME]:
        r, c = divmod(i, 9)
        cx, cy = 17.5 + 30.625 * c, 17.5 + 31.111 * r
        x0, y0 = max(int(cx - 20), 0), max(int(cy - 20), 0)
        x1, y1 = min(int(cx + 20), 280), min(int(cy + 20), 315)
        crop = Image.fromarray(img315[y0:y1, x0:x1]).resize((40, 40), Image.LANCZOS)
        out.append(np.array(crop))
        labs.append(int(labels[i]))
    return out, labs


def main():
    all_c, all_l, all_s = [], [], []
    for pool in POOLS:
        p = HERE / f"data/{pool}.npz"
        if not p.exists():
            print(f"[跳过] {pool}")
            continue
        d = np.load(str(p), allow_pickle=True)
        imgs, labels = d["imgs"], d["labels"]
        n = len(imgs)
        for k in range(n):
            cs, ls = crop_frame(imgs[k], labels[k])
            all_c += cs
            all_l += ls
            all_s += [pool] * len(cs)
        print(f"{pool}: {n} 帧 → 累计 {len(all_c)} 格", flush=True)
        del d, imgs, labels
    cells = np.stack(all_c)
    np.savez_compressed(HERE / "data/cell_bank_lo.npz", cells=cells,
                        labels=np.array(all_l, np.int64), src=np.array(all_s))
    u, cnt = np.unique(all_l, return_counts=True)
    print(f"产出 data/cell_bank_lo.npz: {cells.shape}, 类分布 {dict(zip(u.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()
