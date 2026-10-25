# -*- coding: utf-8 -*-
"""26 帧评测格原生高清重抠（v63 第 1 波）。

eval npz 只有 315×280 warp 定稿位图（格 31.5px）；cell_cls 的核心卖点是原生高清格
（App 帧源图 1080×2280 格约 100px）。本脚本从源图重走链路：
  RTMPose 角点 → ×2 高清 warp（900×1000，格 87.5px）→ 抠格 88px → 128。

质检闸（关键）：HD warp 与 npz 定稿帧对齐相关 <0.80 的帧弃用高清、回退上采样低清格
（防角点检测错位导致格错位——评测格错位=全部结论作废）。

产出：data/eval_cells_hd.npz（cells[26,90,128,128,3] + labels + files + src + quads.json）
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import pose_corner as PC

HERE = Path(__file__).resolve().parent
SRC_DIRS = [HERE / "data/photo_crawl/raw", HERE / "data/photo_crawl/video_frames",
            HERE / "data/photo_crawl_eval/raw"]
GRID_DST_HD = np.array([[100, 100], [800, 100], [100, 900], [800, 900]], np.float32)  # ×2 网格角
OUT = HERE / "data/eval_cells_hd.npz"


def find_src(fname: str) -> Path | None:
    for d in SRC_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None


def main():
    d = np.load(HERE / "data/eval_photo_human_v2.npz", allow_pickle=True)
    files, labels, ref_imgs = list(d["files"]), d["labels"], d["imgs"]
    det = PC.PoseCornerDetector()
    cells = np.zeros((26, 90, 128, 128, 3), np.uint8)
    quads, srcs, mode = [], [], []
    for f, fname in enumerate(files):
        p = find_src(fname)
        assert p is not None, f"源图缺失: {fname}"
        src = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)  # 中文路径兼容
        src = cv2.resize(src, (1600, int(src.shape[0] * 1600 / src.shape[1]))
                         ) if max(src.shape[:2]) > 1600 else src  # 过大图缩边（RTMPose 仿射友好）
        corners = det.detect_with_check(src)
        if corners is None:
            print(f"f{f:02d} {fname}: 角点不可信 → 低清回退")
            corners = None
        else:
            # 对齐质检：标准 warp450 → cls 视图与 npz 定稿帧比对
            H = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                            np.array([[50, 50], [400, 50], [50, 450], [400, 450]], np.float32))
            w450 = cv2.warpPerspective(src, H, (450, 500), flags=cv2.INTER_LINEAR)
            cls_view = cv2.cvtColor(cv2.resize(w450[25:475, 25:425], (280, 315)), cv2.COLOR_BGR2RGB)
            corr = np.corrcoef(cls_view.astype(np.float32).ravel(),
                               ref_imgs[f].astype(np.float32).ravel())[0, 1]
            print(f"f{f:02d} {fname}: 对齐相关 {corr:.3f}", end="")
            if corr < 0.80:
                print(" → 弃用（错位风险）")
                corners = None
            else:
                print(" ✓")
        if corners is not None:  # 高清抠格
            H = cv2.getPerspectiveTransform(corners.astype(np.float32), GRID_DST_HD)
            hd = cv2.warpPerspective(src, H, (900, 1000), flags=cv2.INTER_LINEAR)
            for r in range(10):
                for c in range(9):
                    cx, cy = 100 + 87.5 * c, 100 + 88.888 * r
                    crop = hd[int(cy - 44):int(cy + 44), int(cx - 44):int(cx + 44)]
                    cells[f, r * 9 + c] = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2RGB)
            quads.append(corners.tolist())
            mode.append("hd")
        else:  # 低清回退：npz 定稿帧抠格
            from cell_eval import crop_cells
            cc, _ = crop_cells(ref_imgs[f], labels[f])
            cells[f] = cc
            quads.append(None)
            mode.append("lo")
        srcs.append(p.name)
    np.savez_compressed(OUT, cells=cells, labels=labels, files=np.array(files),
                        src=np.array(srcs), mode=np.array(mode))
    json.dump(quads, open(HERE / "data/eval_cells_hd_quads.json", "w"))
    n_hd = sum(m == "hd" for m in mode)
    print(f"\n产出 {OUT.name}: 高清 {n_hd}/26 帧，低清回退 {26 - n_hd} 帧")


if __name__ == "__main__":
    main()
