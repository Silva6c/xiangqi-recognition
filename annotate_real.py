# -*- coding: utf-8 -*-
"""真实帧自动标注（v3 数据飞轮第 2 步）。

链路：打谱截屏帧 → RTMPose 角点（端上同款移植）→ warp450 → 中心裁剪/缩放 280x315
→ nano_v3 分类（bench 已证其在 cnvcs 域 ~99.7% 准，做训练标签足够）→ 逐格置信度过滤。

输出 data/real_frames_labeled.npz：
- imgs:  [N,280,315,3] uint8 RGB（v3 混训直接用）
- labels:[N,90] int64（低置信格 = IGNORE_LABEL=-100，训练时忽略）
- bconf: [N] 每帧平均最大 logit（整帧质量分）
- files: [N] 源文件名
- meta:  json 字符串（采集参数/阈值/统计）

诊断模式（--diag）只打印 pose 成功率与置信度分布，不落盘。
用法：python annotate_real.py [--cell-thr 2.0] [--board-thr 4.0] [--diag]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import pose_corner as PC

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
FRAMES_DIR = HERE / "data/real_frames"
OUT_NPZ = HERE / "data/real_frames_labeled.npz"

IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

# 多域标注（bench 失真复刻）：1.0 = RTMPose 原生域；bench 型 = bench 10 帧 quads 实测
# 的框-网格失真——框比网格宽（横 951.8 vs 925、纵 1065 vs 1060）且中心右下偏 (+23.8,+9.5)
# 原图px，等效 corners 相对内容**放大 (1.029,1.005)+平移**，内容在 warp 域缩小并左上偏。
# 注意 v5 的 (0.95,0.98) 收缩方向反了（收缩=内容放大），bench 内容缩小域从未进训练——本版修正。
# ≥2 域标签一致才采信；各域图像全部进训练集。
SCALE_CORRECTIONS = (((1.0, 1.0), (0.0, 0.0)), ((1.029, 1.005), (23.8, 9.5)))


def shrink_corners(corners: np.ndarray, spec) -> np.ndarray:
    """角点仿射：scale=(sx,sy) 朝质心缩放 + shift=(dx,dy) 平移（原图 px）。
    scale>1 = 框外扩 = warp 内容缩小；shift 与 bench 框-网格中心偏差同向。"""
    (sx, sy), (dx, dy) = spec
    c = corners.mean(axis=0, keepdims=True)
    out = c + (corners - c)
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out + np.array([dx, dy], np.float32)


def warp450_to_cls(warp450_bgr: np.ndarray) -> np.ndarray:
    """warp450 → 280x315 RGB uint8（与端上 OnnxGeom 预处理同契约：中心裁剪 400x450 → resize）"""
    crop = warp450_bgr[25:475, 25:425]              # H=450, W=400
    img = cv2.resize(crop, (280, 315), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-thr", type=float, default=0.5,
                    help="逐格最大概率低于该值 → 该格标签置 IGNORE（nano_v3 输出为概率尺度）")
    ap.add_argument("--board-thr", type=float, default=0.6,
                    help="整帧平均概率低于该值 → 整帧丢弃")
    ap.add_argument("--diag", action="store_true", help="仅诊断不落盘")
    args = ap.parse_args()

    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    print(f"待标注 {len(frames)} 帧")

    pose = PC.PoseCornerDetector()
    cls = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    cls_in = cls.get_inputs()[0].name

    imgs, labels_all, bconfs, files = [], [], [], []
    pose_fail: list[str] = []
    all_cell_conf: list[float] = []
    all_board_conf: list[float] = []
    agree_flags: list[np.ndarray] = []
    seen_boards: set[bytes] = set()
    n_dup = 0

    for f in frames:
        frame = cv2.imread(str(f))
        if frame is None:
            pose_fail.append(f.name + ":read_fail")
            continue
        corners = pose.detect_with_check(frame)
        if corners is None:
            pose_fail.append(f.name + ":pose_fail")
            continue

        # 多域推理：各域 warp → 280x315 → nano_v3；≥2 域预测一致才采信该格
        domain_imgs, domain_preds, domain_conf = [], [], []
        for sc in SCALE_CORRECTIONS:
            warp = PC.warp_frame_to_warp450(frame, shrink_corners(corners, sc))
            cls_img = warp450_to_cls(warp)
            x = cls_img.astype(np.float32).transpose(2, 0, 1) / 255.0
            x = np.ascontiguousarray((x - IM_MEAN) / IM_STD)[None]
            logits = cls.run(None, {cls_in: x})[0].reshape(90, 16)
            domain_imgs.append(cls_img)
            domain_preds.append(logits.argmax(axis=1))
            domain_conf.append(logits.max(axis=1))

        stack = np.stack(domain_preds)                      # [D,90]
        mode = np.zeros(90, np.int64)
        votes = np.zeros(90, np.int64)
        for j in range(90):
            cnt = np.bincount(stack[:, j], minlength=16)
            mode[j] = cnt.argmax()
            votes[j] = cnt.max()
        agree = votes >= 2
        agree_flags.append(agree)
        pred = mode
        cell_conf = np.min(np.stack(domain_conf), axis=0).copy()   # 各域最低置信
        cell_conf[~agree] = 0.0                             # 不一致格按零置信（→IGNORE）
        bconf = float(cell_conf.mean())
        all_cell_conf.extend(cell_conf.tolist())
        all_board_conf.append(bconf)

        # 局面去重（按双域一致后的预测序列）
        key = pred.tobytes()
        if key in seen_boards:
            n_dup += 1
            continue
        seen_boards.add(key)

        if not args.diag:
            labels = pred.astype(np.int64).copy()
            labels[~agree] = -100                       # 双域不一致格：不采信
            labels[cell_conf < args.cell_thr] = -100    # 低置信格忽略
            if bconf < args.board_thr:                  # 整帧质量分过低
                continue
            for dimg in domain_imgs:                    # 两域图像各成一个训练样本
                imgs.append(dimg)
                labels_all.append(labels)
                bconfs.append(bconf)
                files.append(f.name)

    print(f"\npose 失败 {len(pose_fail)} 帧: {pose_fail[:8]}{'...' if len(pose_fail) > 8 else ''}")
    print(f"局面去重剔除 {n_dup} 帧")
    ag = np.concatenate(agree_flags) if agree_flags else np.array([True])
    print(f"多域{[s for s in SCALE_CORRECTIONS]}标签一致率(≥2票): {ag.mean() * 100:.3f}%（不一致格已置 IGNORE）")
    q = np.quantile(all_cell_conf, [0.01, 0.05, 0.5, 0.95, 0.99])
    print(f"逐格 logit 分位: p1={q[0]:.2f} p5={q[1]:.2f} p50={q[2]:.2f} p95={q[3]:.2f} p99={q[4]:.2f}")
    qb = np.quantile(all_board_conf, [0.01, 0.05, 0.5, 0.95])
    print(f"整帧平均 logit 分位: p1={qb[0]:.2f} p5={qb[1]:.2f} p50={qb[2]:.2f} p95={qb[3]:.2f}")
    print(f"cell<{args.cell_thr} 占比 {np.mean(np.array(all_cell_conf) < args.cell_thr) * 100:.2f}%"
          f" | board<{args.board_thr} 占比 {np.mean(np.array(all_board_conf) < args.board_thr) * 100:.2f}%")

    if args.diag:
        return

    z = np.savez_compressed(
        OUT_NPZ,
        imgs=np.stack(imgs),
        labels=np.stack(labels_all),
        bconf=np.array(bconfs, np.float32),
        files=np.array(files),
        meta=json.dumps({
            "cell_thr": args.cell_thr, "board_thr": args.board_thr,
            "source": "cnvcs 真实帧两批：frame_01xx=打谱·碧桂园杯/古谱逐局推进, "
                      "frame_02xx~03xx=打谱·古谱八虎征西/行军用兵, frame_0001=残局测试, "
                      "frame_04xx=对弈模式（提示自动走红+特级大师应黑, 与 bench 同界面域）",
            "annotator": "nano_v3-0319（bench 同口径 ~99.7% 准）",
        }, ensure_ascii=False),
    )
    print(f"\n落盘 {OUT_NPZ}: {len(imgs)} 帧（IGNORE 格占比 "
          f"{np.mean(np.concatenate(labels_all) == -100) * 100:.2f}%）")


if __name__ == "__main__":
    main()
