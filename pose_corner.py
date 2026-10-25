# -*- coding: utf-8 -*-
"""RTMPose 角点定位的 Python 移植（评测用：为 bench 底片等真实帧产分类输入）。

与端上 OnnxBoardRecognizer.detectCorners / poseAffine / OnnxGeom.simccKeypoints
逐项对齐（移植前逐行核对 Kotlin 源）：
- poseAffine：全图 bbox × 1.25 padding，三分点法（rot=0），cv2.getAffineTransform 等价 DLT；
- SimCC 解码：argmax over 2×inputSize → 坐标 = argmax / half × inputSize；
- 角点顺序：[TL, TR, BL, BR]（A0 左上 / A8 右上 / J0 左下 / J8 右下）。

仅用于离线评测（给真实帧产 280×315 分类输入），不参与训练。
许可证：本文件为独立再实现（读 Kotlin 语义后重写），非代码复制。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

POSE_INPUT = 256
POSE_PADDING = 1.25
DEFAULT_MODEL = Path(__file__).parents[2] / "app/src/main/assets/models/4_v6-0301.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def pose_affine(w: int, h: int) -> np.ndarray:
    """全图 bbox + 1.25 padding 的仿射矩阵（原图 → 256×256），语义同端上 poseAffine"""
    cx, cy = w / 2.0, h / 2.0
    src_w = w * POSE_PADDING
    src_dir_x = -0.5 * src_w
    dst_w = dst_h = float(POSE_INPUT)
    dst_dir_x = -0.5 * dst_w
    # 三组源点：center / center+dir / 第三点 = p1 + (dir.y, -dir.x) = p1 + (0, -srcDirX)
    src = np.array([[cx, cy], [cx + src_dir_x, cy], [cx + src_dir_x, cy - src_dir_x]], np.float32)
    dst = np.array([[dst_w / 2, dst_h / 2],
                    [dst_w / 2 + dst_dir_x, dst_h / 2],
                    [dst_w / 2 + dst_dir_x, dst_h / 2 - dst_dir_x]], np.float32)
    return cv2.getAffineTransform(src, dst)


def corner_affine(w: int, h: int) -> np.ndarray:
    """自研角点模型（v64）输入仿射：长边方形窗（短边 pad 黑）→ 256×256。

    与 pose_affine（帧宽×1.25 中心方形）的差异与动机：竖屏 1080×2280 下旧窗仅覆盖
    y∈[465,1815]，bh≥1350px 的棋盘无论居中与否必然部分出窗（screen30 首考 30 帧中
    27 帧半盘可见 = RTMPose 57% 的主因；SimCC bins [0,512] 无法表达出窗角点）。
    长边窗全帧必然覆盖、无拉伸；端上 OnnxBoardRecognizer.poseAffine 同步此语义。
    """
    L = float(max(w, h))
    cx, cy = w / 2.0, h / 2.0
    d = float(POSE_INPUT)
    src = np.array([[cx, cy], [cx - L / 2, cy], [cx - L / 2, cy - L / 2]], np.float32)
    dst = np.array([[d / 2, d / 2], [0, d / 2], [0, 0]], np.float32)
    return cv2.getAffineTransform(src, dst)


class PoseCornerDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL):
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        # 输出两个张量 simcc_x / simcc_y（按名字找，避免顺序假设）
        names = [o.name for o in self.sess.get_outputs()]
        self.x_name = next(n for n in names if "x" in n.lower())
        self.y_name = next(n for n in names if "y" in n.lower())

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """帧 BGR → 网格四角 [4,2]（TL,TR,BL,BR 原图坐标）；不可信返回 None"""
        h, w = frame_bgr.shape[:2]
        m = pose_affine(w, h)
        warped = cv2.warpAffine(frame_bgr, m, (POSE_INPUT, POSE_INPUT))
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1) - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
        outs = self.sess.run(None, {self.iname: x[None].astype(np.float32)})
        sx, sy = outs[0][0], outs[1][0]  # [4, 512]
        half = sx.shape[1]
        kps = np.stack([sx.argmax(axis=1) / half * POSE_INPUT,
                        sy.argmax(axis=1) / half * POSE_INPUT], axis=1).astype(np.float32)
        # 逆仿射回原图
        inv = cv2.invertAffineTransform(m)
        ones = np.concatenate([kps, np.ones((4, 1), np.float32)], axis=1)
        return (inv @ ones.T).T  # [4,2]

    def detect_with_check(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """detect + 端上同款 boxPlausible 校验（面积占比 2%~120%、宽高比 0.5~2.2）"""
        corners = self.detect(frame_bgr)
        if corners is None:
            return None
        x0, y0 = corners.min(axis=0); x1, y1 = corners.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return None
        if not (0.5 <= bw / bh <= 2.2):
            return None
        area = bw * bh / (w := frame_bgr.shape[1]) / (h := frame_bgr.shape[0])
        if not (0.02 <= area <= 1.2):
            return None
        return corners

def warp_frame_to_warp450(frame_bgr: np.ndarray, corners_tl_tr_bl_br: np.ndarray) -> np.ndarray:
    """角点(TL,TR,BL,BR) → warp450 BGR（与端上 warpBoard 同义）"""
    GRID_DST = np.array([[50, 50], [400, 50], [50, 450], [400, 450]], np.float32)  # TL,TR,BL,BR
    H = cv2.getPerspectiveTransform(corners_tl_tr_bl_br.astype(np.float32), GRID_DST)
    return cv2.warpPerspective(frame_bgr, H, (450, 500),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
