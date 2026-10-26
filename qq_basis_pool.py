# -*- coding: utf-8 -*-
"""QQ 事故截图基准池制作（v67 判据：QQ 象棋截图定位全对，不入库临时脚本）。

GT = v4 定位结果（v4 全链路 89/90 目检实证其在该截图定位正确）+ 目检可视化复核。
输出 phone/.cache/corner_qq_basis.npz（files + corners 模型序 [TL,TR,BL,BR]）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent          # recog 目录
CACHE = HERE / "data"                                      # 数据缓存根（不入库）
sys.path.insert(0, str(HERE))
from cell_model import DWBlk  # noqa: E402
from pose_corner import IMAGENET_MEAN, IMAGENET_STD, corner_affine  # noqa: E402

IMG = Path(r"C:\Users\12567\Downloads\Screenshot_20261026_060632_com.tencent.qqgame.xq.jpg")
CORNER_IN = 256


class CornerSimCC_v64(nn.Module):
    """v64 旧结构复刻（4 角点 SimCC 头），专用于读旧权重做基准 GT。"""

    def __init__(self, width=(48, 64, 96, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width[0]), nn.ReLU(inplace=True))
        strides = (2, 2, 1, 1, 1)
        self.blocks = nn.Sequential(*[
            DWBlk(width[i], width[i + 1], strides[i]) for i in range(len(strides))])
        self.proj = nn.Conv2d(width[-1], 4, 1, bias=False)
        self.pool_x = nn.Conv2d(4, 4, (32, 1), stride=(32, 1), groups=4, bias=False)
        self.pool_y = nn.Conv2d(4, 4, (1, 32), stride=(1, 32), groups=4, bias=False)
        self.fc_x = nn.Linear(4 * 32, 4 * 512)
        self.fc_y = nn.Linear(4 * 32, 4 * 512)

    def forward(self, x):
        f = self.blocks(self.stem(x))
        t = self.proj(f)
        gx = self.pool_x(t).flatten(2)
        gy = self.pool_y(t).flatten(2)
        b = x.shape[0]
        return (self.fc_x(gx.flatten(1)).view(b, 4, 512),
                self.fc_y(gy.flatten(1)).view(b, 4, 512))


def v4_detect(model, img_bgr):
    """v4 旧契约推理：输出 (simcc_x, simcc_y) 各 [1,4,512]，argmax/512×256。"""
    h, w = img_bgr.shape[:2]
    M = corner_affine(w, h)
    warped = cv2.warpAffine(img_bgr, M, (CORNER_IN, CORNER_IN),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb.transpose(2, 0, 1) - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
    with torch.no_grad():
        hx, hy = model(torch.from_numpy(x[None]))
    kps = np.stack([hx[0].numpy().argmax(1) / 512 * CORNER_IN,
                    hy[0].numpy().argmax(1) / 512 * CORNER_IN], 1).astype(np.float32)
    inv = cv2.invertAffineTransform(M)
    return (inv @ np.concatenate([kps, np.ones((4, 1), np.float32)], 1).T).T  # 原图 [4,2]


def main():
    img = cv2.imdecode(np.fromfile(str(IMG), dtype=np.uint8), cv2.IMREAD_COLOR)
    print(f"截图 {img.shape[1]}x{img.shape[0]}")
    model = CornerSimCC_v64()
    model.load_state_dict(torch.load(str(HERE / "runs/corner_v4/best.pt"),
                                     map_location="cpu"))
    model.eval()
    det = v4_detect(model, img)
    print("v4 四角 [TL,TR,BL,BR]:", det.astype(int).tolist())

    # 目检可视化（四角+框画原图，缩半存）
    vis = img.copy()
    cv2.polylines(vis, [det.astype(np.int32)], True, (0, 0, 255), 6)
    for px, py in det:
        cv2.circle(vis, (int(px), int(py)), 14, (0, 255, 0), -1)
    cv2.imwrite(str(CACHE / "qq_basis_check.png"), cv2.resize(vis, None, fx=0.4, fy=0.4))

    np.savez(str(CACHE / "corner_qq_basis.npz"),
             files=np.array([str(IMG)]), corners=det[None].astype(np.float32),
             domain=np.array(["qq_ornate"]), split=np.array(["holdout"]))
    print("已存 data/corner_qq_basis.npz + data/reports/qq_basis_check.png（目检：绿点须落格网交叉点）")


if __name__ == "__main__":
    main()
