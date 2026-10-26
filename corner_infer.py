# -*- coding: utf-8 -*-
"""自研角点模型推理孪生（v68 混合头）：与端上 OnnxBoardRecognizer.detectCorners 同构。

链路：corner_affine（长边方形窗）→ 256×256 → ImageNet 归一化 → 模型 →
解码（双头各 [7,512] argmax/域换算 → box6+res8 → 端点 = c±u±v + clamp(res,±RES_LIM)）
→ 逆仿射回原图。护栏（替代旧 boxPlausible，四边一致性）：对边平行由 box 构造保证、
残差限幅 8px@256 不足以破坏平行性，数值校验边长下限（塌缩拒绝）、宽高比、面积。
支持 torch state_dict 与 onnx 双后端（导出后对拍逐位验证用）。
仅离线评测用，不参与训练。
"""
from pathlib import Path

import cv2
import numpy as np

from corner_model import (BOX_LO, BOX_HI, RES_LIM, CORNER_IN, NPK, SIMCC_BINS,
                          X_CH, Y_CH, CornerSimCC, quad_to_gt)
from pose_corner import IMAGENET_MEAN, IMAGENET_STD, corner_affine

# 14 维值域（box6 + res8）
ALL_LO = np.array(list(BOX_LO) + [-RES_LIM] * 8, np.float32)
ALL_HI = np.array(list(BOX_HI) + [RES_LIM] * 8, np.float32)


def corners_from_parts_np(gt14):
    """14 维参数（numpy [...,14]）→ 四角 [TL,TR,BL,BR]（[...,4,2]）。"""
    gt14 = np.asarray(gt14, np.float32)
    box, res8 = gt14[..., :NPK], gt14[..., NPK:]
    cx, cy, ux, uy, vx, vy = [box[..., i] for i in range(6)]
    r = np.clip(res8, -RES_LIM, RES_LIM)
    pts = np.stack([
        np.stack([cx - ux - vx + r[..., 0], cy - uy - vy + r[..., 1]], -1),   # TL
        np.stack([cx + ux - vx + r[..., 2], cy + uy - vy + r[..., 3]], -1),   # TR
        np.stack([cx - ux + vx + r[..., 4], cy - uy + vy + r[..., 5]], -1),   # BL
        np.stack([cx + ux + vx + r[..., 6], cy + uy + vy + r[..., 7]], -1)], -2)  # BR
    return pts.astype(np.float32)


class CornerInfer:
    """detect/detect_with_check 与 pose_corner.PoseCornerDetector 同接口
    （角点 [TL,TR,BL,BR] 原图坐标；护栏参数对齐旧 boxPlausible：宽高比 0.5~2.2、
    面积 2%~120%）。"""

    def __init__(self, torch_weights: Path | None = None, onnx_path: Path | None = None):
        self.sess = None
        self.model = None
        if torch_weights is not None and str(torch_weights).endswith(".onnx"):
            onnx_path, torch_weights = torch_weights, None   # 按后缀自动分流（易用性）
        if onnx_path is not None:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
        else:
            self.model = CornerSimCC()
            self.model.load_state_dict(np.load(str(torch_weights), allow_pickle=True)
                                       if str(torch_weights).endswith(".npy") else
                                       __import__("torch").load(str(torch_weights),
                                                                map_location="cpu"))
            self.model.eval()

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.sess is not None:
            outs = self.sess.run(None, {self.iname: x[None].astype(np.float32)})
            return outs[0][0], outs[1][0]      # [7,512] 双头（导出固化 x 前 y 后）
        import torch
        with torch.no_grad():
            hx, hy = self.model(torch.from_numpy(x[None]))
        return hx[0].numpy(), hy[0].numpy()

    def _decode_auto(self, hx: np.ndarray, hy: np.ndarray) -> np.ndarray:
        """三代契约自动分流（按每头分布数）：4=v64 直回归、3=v67 box、7=v68 混合。

        发布权重 corner_simcc_v1.onnx 是 v4（[4,512]），实验权重是 v67/v68——
        评测链必须能跑全部三代（baseline 复现与实验对拍同工具）。
        """
        n = hx.shape[0]
        if n == 4:   # v64 直回归：4 角点各自 (x,y) 分布，argmax/512×256
            return np.stack([hx.argmax(1) / SIMCC_BINS * CORNER_IN,
                             hy.argmax(1) / SIMCC_BINS * CORNER_IN], 1).astype(np.float32)
        gt14 = np.zeros(NPK + 8, np.float32)
        if n == 3:   # v67 box：x 头 (cx,ux,vx)，y 头 (cy,uy,vy)，残差恒 0
            for j, p in enumerate((0, 2, 4)):
                gt14[p] = hx[j].argmax() / SIMCC_BINS * (ALL_HI[p] - ALL_LO[p]) + ALL_LO[p]
            for j, p in enumerate((1, 3, 5)):
                gt14[p] = hy[j].argmax() / SIMCC_BINS * (ALL_HI[p] - ALL_LO[p]) + ALL_LO[p]
        elif n == 7:  # v68 混合：box6 + 残差8（X_CH/Y_CH 全局序）
            for j, p in enumerate(X_CH):
                gt14[p] = hx[j].argmax() / SIMCC_BINS * (ALL_HI[p] - ALL_LO[p]) + ALL_LO[p]
            for j, p in enumerate(Y_CH):
                gt14[p] = hy[j].argmax() / SIMCC_BINS * (ALL_HI[p] - ALL_LO[p]) + ALL_LO[p]
        else:
            raise ValueError(f"未知角点模型契约：每头 {n} 分布（期望 4/3/7）")
        return corners_from_parts_np(gt14)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        M = corner_affine(w, h)
        warped = cv2.warpAffine(frame_bgr, M, (CORNER_IN, CORNER_IN),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1) - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
        hx, hy = self._run(x)
        kps = self._decode_auto(hx, hy)                  # [4,2] 256 空间
        inv = cv2.invertAffineTransform(M)
        ones = np.concatenate([kps, np.ones((4, 1), np.float32)], axis=1)
        return (inv @ ones.T).T  # [4,2] 原图坐标

    def detect_with_check(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        corners = self.detect(frame_bgr)
        if corners is None:
            return None
        # v68 护栏（替代旧 boxPlausible 的 bbox 检查）：box 构造 + 限幅残差下对边
        # 近似平行（残差 ≤8px@256 不足以造出简并），护栏校验边长/比例/面积。
        tl, tr, bl, br = corners
        wvec, hvec = tr - tl, bl - tl            # 四边形两边
        wlen, hlen = float(np.linalg.norm(wvec)), float(np.linalg.norm(hvec))
        if wlen < 8 or hlen < 8:                 # 塌缩拒绝（<8px@原图）
            return None
        if not (0.5 <= wlen / hlen <= 2.2):      # 宽高比（旧 boxPlausible 同域）
            return None
        h, w = frame_bgr.shape[:2]
        area = abs(float(np.cross(wvec, hvec)))
        if not (0.02 <= area / (w * h) <= 1.2):  # 面积占比（旧 boxPlausible 同域）
            return None
        return corners
