# -*- coding: utf-8 -*-
"""真实域棋子移植增广（v12 数据飞轮：外观域差的最后一公里）。

bench 残余错格的头号根因是"真实帧库 0 覆盖的组合"——沉底黑炮 (9,0)/(9,1) 占
v10a 残余 19 错格的 42%，而 212 张真实帧里 (9,0) 只有 '.'/'R'。合成渲染的沉底
炮外观（v10b 定向 8% + 蒸馏）已证明救不动：真实棋子是浮雕圆片+光照渐变，合成
平面绘制不像。

本脚本做"同帧棋子移植"：从真实帧里裁下真棋子 patch（如帧内某格的黑炮），
羽化后贴到同帧的目标空位（如 (9,0)）——同帧光照/主题/失真完全一致，痕迹最小；
标签同步改目标格。teacher（nano_v3）对移植帧推理做自动质检：teacher 在 bench
域 100%（=画面事实），它认得贴片格 → 贴片成功保留；不认得 → 丢弃。

产物 real_frames_transplanted.npz：imgs [M,280,315,3] u8 RGB + labels [M,90]。
用法：python augment_real_transplant.py [--out data/real_frames_transplanted.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import renderer as R
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"

CLS = ".xKABNRCPkabnrcp"

# 移植组合表：(源类别, 目标格位列表)。头号目标是 v10a 残余错格的热点空位。
# 源类别在帧内找同色棋子 patch；目标格位必须当前为空（label='.'）。
# v18 扩充：v17 bench 残余 18 错格的靶点（背排 r/n 互认、(9,7) N/R、(9,6)B、(8,5)p）。
TRANSPLANTS = [
    ("c", [(9, 0), (9, 1), (9, 7), (9, 8)]),   # 沉底黑炮（头号根因）
    ("n", [(0, 1), (0, 7), (9, 0), (9, 8), (9, 7)]),  # 顶/底行黑马（n→r 混淆）
    ("p", [(9, 0), (9, 5), (9, 8), (6, 6), (8, 5)]),  # 黑卒零星混淆位
    ("r", [(9, 8), (0, 1), (0, 7)]),           # 黑车（r→n 混淆位，v18 加 (0,7)）
    ("R", [(9, 7)]),                           # 红车落 (9,7)（幻视 R 反例）
    ("N", [(9, 7)]),                           # 红马落 (9,7)（N/R 互翻）
    ("B", [(9, 6)]),                           # 红相落 (9,6)（B→p 混淆）
]


def cls_cell_center(i: int, j: int) -> tuple[int, int]:
    """cls 域 (280×315) 的格心：warp450 域格心 (50+j*43.75, 50+i*44.444) 经
    中心裁剪 400×450(offset 25,25) + resize 280×315（×0.7）"""
    cx = (50 + j * 43.75 - 25) * 0.7
    cy = (50 + i * 44.444 - 25) * 0.7
    return int(round(cx)), int(round(cy))


def circle_patch(img: np.ndarray, cx: int, cy: int, radius: int = 22) -> np.ndarray | None:
    """裁圆形棋子 patch（含 4px 边距），返回 BGR patch 或 None（越界）"""
    h, w = img.shape[:2]
    r = radius + 4
    x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1].copy()


def paste_piece(dst_img: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> None:
    """羽化圆形贴片：圆形 mask（r=patch 半径-2）+ 3px 高斯羽化边缘"""
    ph, pw = patch.shape[:2]
    r = pw // 2 - 4
    mask = np.zeros((ph, pw), np.float32)
    cv2.circle(mask, (pw // 2, ph // 2), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    x0, y0 = cx - pw // 2, cy - ph // 2
    h, w = dst_img.shape[:2]
    gx0, gy0 = max(0, x0), max(0, y0)
    gx1, gy1 = min(w, x0 + pw), min(h, y0 + ph)
    if gx1 <= gx0 or gy1 <= gy0:
        return
    sub = mask[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    roi = dst_img[gy0:gy1, gx0:gx1].astype(np.float32)
    src = patch[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0].astype(np.float32)
    dst_img[gy0:gy1, gx0:gx1] = (src * sub[..., None] + roi * (1 - sub[..., None])).astype(np.uint8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "data/real_frames_labeled.npz"))
    ap.add_argument("--out", default=str(HERE / "data/real_frames_transplanted.npz"))
    ap.add_argument("--min-teacher-conf", type=float, default=0.55,
                    help="teacher 对贴片格的软标签概率下限（质检）")
    args = ap.parse_args()

    z = np.load(args.src, allow_pickle=False)
    imgs, labs = z["imgs"], z["labels"]          # [N,315,280,3] u8 RGB / [N,90]
    n_src = len(imgs)
    print(f"源集：{n_src} 帧真实标注")

    # teacher 质检 session
    import onnxruntime as ort
    sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    out_imgs, out_labs, notes = [], [], []
    n_try = n_pass = 0
    for k in range(n_src):
        img = imgs[k]                            # RGB
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = labs[k]
        for src_cls, targets in TRANSPLANTS:
            src_idx = CLS.index(src_cls)
            src_cells = [i for i in range(90) if lab[i] == src_idx]
            if not src_cells:
                continue
            for (ti, tj) in targets:
                t_idx = ti * 9 + tj
                if lab[t_idx] != CLS.index("."):
                    continue                     # 目标非空，跳过
                # 随机选 2 个源格位各试一次
                for si in np.random.RandomState(k * 131 + t_idx).choice(src_cells, size=min(2, len(src_cells)), replace=False):
                    n_try += 1
                    scx, scy = cls_cell_center(int(si) // 9, int(si) % 9)
                    tcx, tcy = cls_cell_center(ti, tj)
                    patch = circle_patch(bgr, scx, scy)
                    if patch is None:
                        continue
                    patched = bgr.copy()
                    paste_piece(patched, patch, tcx, tcy)
                    new_lab = lab.copy()
                    new_lab[t_idx] = src_idx
                    # teacher 质检：贴片格软标签概率 + 全盘 argmax 是否含新格
                    x = cv2.cvtColor(patched, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
                    x = ((x - T.IM_MEAN_NP) / T.IM_STD_NP)[None].astype(np.float32)
                    lo = sess.run(None, {iname: np.ascontiguousarray(x)})[0][0]  # [90,16]
                    p_row = lo[t_idx].astype(np.float64)
                    p_row = p_row / max(1e-9, p_row.sum())
                    conf = float(p_row[src_idx])
                    if conf < args.min_teacher_conf:
                        continue
                    # 贴片格 argmax 必须命中目标类（软概率之外的硬校验）
                    if int(lo[t_idx].argmax()) != src_idx:
                        continue
                    out_imgs.append(cv2.cvtColor(patched, cv2.COLOR_BGR2RGB))
                    out_labs.append(new_lab)
                    notes.append((k, (ti, tj), src_cls, round(conf, 3)))
                    n_pass += 1
        if (k + 1) % 60 == 0:
            print(f"  {k + 1}/{n_src} 帧，通过 {n_pass}/{n_try}", flush=True)

    M = len(out_imgs)
    print(f"移植完成：尝试 {n_try}，teacher 质检通过 {n_pass}（{n_pass / max(1, n_try):.0%}）")
    np.savez_compressed(args.out, imgs=np.stack(out_imgs).astype(np.uint8),
                        labels=np.stack(out_labs).astype(np.int64))
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"落盘 {args.out}: {M} 样本, {size_mb:.0f}MB")


if __name__ == "__main__":
    main()
