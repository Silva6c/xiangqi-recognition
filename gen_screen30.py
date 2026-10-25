# -*- coding: utf-8 -*-
"""screen30 考卷生成器（v1）：renderer 整屏合成路径首次使用（零泄漏，见设计文档 §二）。

30 帧仿真屏：随机局面/主题/姿态/背景/遮挡，GT=渲染自证（90 格 labels + 网格四角）。
种子 20261025+k（全新，与训练 BASE_SEED 体系无关）。落盘 data/screen30_v1.npz。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import renderer as R

SEED_BASE = 20261025
N = 30
SCREEN = (1080, 2280)
STAGES = ["opening"] * 6 + ["midgame"] * 15 + ["endgame"] * 9


def place_board_fullframe(flat, rng):
    """自控整屏贴放：网格四角 ±14°+透视扰动，保证四角全在 1080×2280 屏内。
    （synth_screen 的中心约束使大占比+旋转帧必然出画且 region 无法包含负坐标角，
    故几何贴放自控；flat 渲染仍全走 renderer=零泄漏。）"""
    import math
    grid_w, grid_h = 350.0, 400.0                      # warp 网格空间 350×400
    bh = SCREEN[1] * rng.uniform(0.55, 0.85)
    bw = bh * grid_w / grid_h
    theta = math.radians(rng.uniform(-14, 14))
    ct, st_ = math.cos(theta), math.sin(theta)
    kst = rng.uniform(0.0, 0.05)
    hw, hh = bw / 2, bh / 2
    quad_rel = np.array([[-hw, -hh], [hw, -hh * (1 - kst)], [hw, hh],
                         [-hw, hh * (1 - kst)]], np.float32)
    rot = np.array([[ct, -st_], [st_, ct]], np.float32)
    quad_rot = quad_rel @ rot.T
    # AABB 缩放保证 ≤92% 屏
    aabb_w = quad_rot[:, 0].max() - quad_rot[:, 0].min()
    aabb_h = quad_rot[:, 1].max() - quad_rot[:, 1].min()
    s = min(1.0, (SCREEN[0] * 0.92) / aabb_w, (SCREEN[1] * 0.92) / aabb_h)
    quad_rel *= s
    hw, hh = hw * s, hh * s
    quad_rot = quad_rel @ rot.T
    aabb_w = quad_rot[:, 0].max() - quad_rot[:, 0].min()
    aabb_h = quad_rot[:, 1].max() - quad_rot[:, 1].min()
    cx0 = rng.uniform(aabb_w / 2 + 10, SCREEN[0] - aabb_w / 2 - 10)
    cy0 = rng.uniform(aabb_h / 2 + 10, SCREEN[1] - aabb_h / 2 - 10)
    quad = quad_rot + np.array([cx0, cy0], np.float32)   # [TL,TR,BR,BL]

    # flat(含 50px 木框) → 画布：网格 src(50,50)-(400,450)×SS 映射 quad
    SS = R.SS
    src_grid = np.array([[50, 50], [400, 50], [400, 450], [50, 450]], np.float32) * SS
    Hm = cv2.getPerspectiveTransform(src_grid, quad.astype(np.float32))

    def apply_h(px_, py_):
        v = Hm @ np.array([px_, py_, 1.0], np.float64)
        return [v[0] / v[2], v[1] / v[2]]

    dst_canvas = np.array([apply_h(0, 0), apply_h(900.0, 0),
                           apply_h(900.0, 1000.0), apply_h(0, 1000.0)], np.float32)
    # flat 像素尺寸 = (WARP_W*SS, WARP_H*SS) = (900, 1000)
    H_canvas = cv2.getPerspectiveTransform(
        np.array([[0, 0], [900.0, 0], [900.0, 1000.0], [0, 1000.0]], np.float32),
        dst_canvas.astype(np.float32))

    # 全屏渐变背景 + warp
    c0 = (rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245))
    c1 = (rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245))
    rows = (np.arange(SCREEN[1], dtype=np.float32) / (SCREEN[1] - 1)).reshape(-1, 1, 1)
    bg = np.repeat((np.array(c0, np.float32) * (1 - rows) +
                    np.array(c1, np.float32) * rows), SCREEN[0], axis=1)
    flat_bgra = cv2.cvtColor(np.array(flat), cv2.COLOR_RGBA2BGRA)
    warped = cv2.warpPerspective(flat_bgra, H_canvas, (SCREEN[0], SCREEN[1]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    out = (bg * (1 - alpha) + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    # 遮挡物（对话框样式，45% 概率 0~2 个，样式对齐 synth_screen）
    for _ in range(rng.randint(0, 2)):
        if rng.random() < 0.45:
            ow = rng.randint(SCREEN[0] // 6, SCREEN[0] // 2)
            oh = rng.randint(SCREEN[1] // 12, SCREEN[1] // 5)
            ox = rng.randint(-ow // 3, max(0, SCREEN[0] - ow // 2))
            oy = rng.randint(0, max(0, SCREEN[1] - oh))
            col = rng.choice([(240, 240, 240), (30, 30, 34), (250, 245, 235), (60, 62, 70)])
            cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), col, -1)
            cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), (120, 120, 120), 2)
    return out, quad


def render_one(k: int):
    """第 k 帧仿真屏。返回 (frame RGB uint8, labels 90, corners TL,TR,BR,BL 绝对坐标)。"""
    rng = random.Random(SEED_BASE + k)
    stage = STAGES[k % len(STAGES)]

    lay = R.random_layout(rng, stage)
    style = R.sample_style(rng, R.load_font_pool())
    flat = R.render_board_flat(lay, style, rng)              # RGBA 900×1000
    canvas_bgr, quad = place_board_fullframe(flat, rng)      # 四角 TL,TR,BR,BL 全在屏内

    labels = np.full(R.ROWS * R.COLS, R.CLASS_INDEX["."], np.int64)
    for (r, c), ch in lay.items():
        labels[r * R.COLS + c] = R.CLASS_INDEX[ch]
    return canvas_bgr[:, :, ::-1], labels, quad, stage


def main():
    imgs, labels, corners, stages, files = [], [], [], [], []
    for k in range(N):
        img, lab, cor, stage = render_one(k)
        imgs.append(img); labels.append(lab); corners.append(cor)
        stages.append(stage); files.append(f"screen30_{k:02d}_{stage}.png")
        n_sub = int(sum(1 for v in lab if v != 0))
        print(f"[{k:02d}] {stage:8s} 子数={n_sub:2d} 角点TL={cor[0].round(0).tolist()}")

    np.savez_compressed(
        str(HERE / "data/screen30_v1.npz"),
        imgs=np.array(imgs, np.uint8), labels=np.array(labels, np.int64),
        corners=np.array(corners, np.float32), stages=np.array(stages),
        files=np.array(files),
        meta=np.array([f"seed_base={SEED_BASE} params_frozen=v63_final "
                       f"(thr0.90/cthr0.60/conflict0.92,0.95)"], np.str_))
    print(f"已落盘 data/screen30_v1.npz：{N} 帧")


if __name__ == "__main__":
    main()
