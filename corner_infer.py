# -*- coding: utf-8 -*-
"""自研角点模型推理孪生（v64）：与端上 OnnxBoardRecognizer.detectCorners 新语义同构。

链路：corner_affine（长边方形窗）→ 256×256 → ImageNet 归一化 → 模型 →
SimCC argmax 解码（argmax/512×256，OnnxGeom.simccKeypoints 同构）→ 逆仿射回原图。
支持 torch state_dict 与 onnx 双后端（导出后对拍逐位验证用）。
仅离线评测用，不参与训练。
"""
from pathlib import Path

import cv2
import numpy as np

from corner_model import CORNER_IN, SIMCC_BINS, CornerSimCC
from pose_corner import IMAGENET_MEAN, IMAGENET_STD, corner_affine


class CornerInfer:
    """detect/detect_with_check 与 pose_corner.PoseCornerDetector 同接口
    （角点 [TL,TR,BL,BR] 原图坐标；boxPlausible 同参数：宽高比 0.5~2.2、面积 2%~120%）。"""

    def __init__(self, torch_weights: Path | None = None, onnx_path: Path | None = None):
        self.sess = None
        self.model = None
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
            return outs[0][0], outs[1][0]      # [4,512] x/y（导出固化 x 前 y 后）
        import torch
        with torch.no_grad():
            lx, ly = self.model(torch.from_numpy(x[None]))
        return lx[0].numpy(), ly[0].numpy()

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        M = corner_affine(w, h)
        warped = cv2.warpAffine(frame_bgr, M, (CORNER_IN, CORNER_IN),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (rgb.transpose(2, 0, 1) - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
        sx, sy = self._run(x)
        kps = np.stack([sx.argmax(axis=1) / SIMCC_BINS * CORNER_IN,
                        sy.argmax(axis=1) / SIMCC_BINS * CORNER_IN], axis=1).astype(np.float32)
        inv = cv2.invertAffineTransform(M)
        ones = np.concatenate([kps, np.ones((4, 1), np.float32)], axis=1)
        return (inv @ ones.T).T  # [4,2] 原图坐标

    def detect_with_check(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        corners = self.detect(frame_bgr)
        if corners is None:
            return None
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return None
        if not (0.5 <= bw / bh <= 2.2):
            return None
        h, w = frame_bgr.shape[:2]
        if not (0.02 <= bw * bh / (w * h) <= 1.2):
            return None
        return corners
