# -*- coding: utf-8 -*-
"""自研角点定位模型（v64）：SimCC 复刻，替换外部 RTMPose（4_v6-0301.onnx，CCH 上游）。

接口契约与 4_v6-0301.onnx 逐项对齐（端上 OnnxBoardRecognizer 换文件+poseAffine 微改）：
- 输入 [1,3,256,256] float CHW，ImageNet 归一化；图为长边方形窗（短边 pad 黑，
  corner_affine v64 语义——旧 1.25 宽窗在竖屏大棋盘下必然半盘出窗，见 pose_corner.py）；
- 输出按序 (simcc_x, simcc_y) 各 [1,4,512]，解码 = argmax/512×256
  （OnnxGeom.simccKeypoints 零改动，Kotlin 按输出顺序取 [0][1]，导出时固化 x 前 y 后）；
- 角点顺序 [TL,TR,BL,BR] = A0/A8/J0/J8 网格交叉点。

架构（冒烟教训 v1 GAP+fc 空间信息全丢→收敛慢，v2 改空间感知头）：
  stem s2 + 4 段 DWBlk 下采到 16×16 + 末段 s1 提深度 → 1×1 proj 到 4 通道角点粗激活图
  → 分组 conv 行/列池化（可学习权重）得 x/y 16 bin 剖面 → fc 内插到 512 bins。
  算子全守 export.py 白名单（Conv/Gemm/Relu/BN/MatMul/Transpose/Reshape）。
设计约束：fp32 ~1.5MB（小模型 int8 已证伪：cell_cls int8 反慢先例，不量化）。
"""
import torch
import torch.nn as nn

from cell_model import DWBlk

CORNER_IN = 256   # 输入边长（端上 POSE_INPUT）
NKP = 4           # 角点数
SIMCC_BINS = 512  # = 2×输入边长（RTMPose 解码契约：argmax/2 = 像素）
FEAT = 32         # 空间特征图边长（32×32 → x/y 剖面 32 bin，亚 bin 由 fc 内插）


class CornerSimCC(nn.Module):
    """256×256×3 → (simcc_x [B,4,512], simcc_y [B,4,512])。"""

    def __init__(self, width=(48, 64, 96, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width[0]), nn.ReLU(inplace=True))
        strides = (2, 2, 1, 1, 1)  # 128→64→32→32→32→32（保 32×32 空间，剖面 32 bin）
        self.blocks = nn.Sequential(*[
            DWBlk(width[i], width[i + 1], strides[i]) for i in range(len(strides))])
        self.proj = nn.Conv2d(width[-1], NKP, 1, bias=False)      # 角点粗激活图 [B,4,16,16]
        # 行/列可学习池化（分组 conv，stride=kernel=整轴）→ x/y 剖面 [B,4,16]
        self.pool_x = nn.Conv2d(NKP, NKP, (FEAT, 1), stride=(FEAT, 1), groups=NKP, bias=False)
        self.pool_y = nn.Conv2d(NKP, NKP, (1, FEAT), stride=(1, FEAT), groups=NKP, bias=False)
        nn.init.constant_(self.pool_x.weight, 1.0 / FEAT)
        nn.init.constant_(self.pool_y.weight, 1.0 / FEAT)
        self.fc_x = nn.Linear(NKP * FEAT, NKP * SIMCC_BINS)
        self.fc_y = nn.Linear(NKP * FEAT, NKP * SIMCC_BINS)

    def forward(self, x):
        f = self.blocks(self.stem(x))          # [B,C,16,16]
        t = self.proj(f)                       # [B,4,16,16]
        gx = self.pool_x(t).flatten(2)         # [B,4,16] 行池化 → x 剖面
        gy = self.pool_y(t).flatten(2)         # [B,4,16] 列池化 → y 剖面
        b = x.shape[0]
        return (self.fc_x(gx.flatten(1)).view(b, NKP, SIMCC_BINS),
                self.fc_y(gy.flatten(1)).view(b, NKP, SIMCC_BINS))


if __name__ == "__main__":
    m = CornerSimCC()
    n = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n/1e3:.1f}K（fp32 ≈ {n*4/1e6:.2f}MB）")
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, CORNER_IN, CORNER_IN))
    print("前向:", y[0].shape, y[1].shape)
