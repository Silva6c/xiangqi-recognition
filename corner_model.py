# -*- coding: utf-8 -*-
"""自研角点定位模型：v67 box 参数化 → v68 混合头（box 结构 + 角点残差）。

v67 教训（docs/训练工作台账.md 归因四条）：纯 box6（c±u±v）端点误差三参数传播
致精度 3 倍差（screen30 对称距离 80.5px vs v4 直回归 14.8px）；且平行四边形假设
与强透视相克（闸剔强透视帧→街拍定位失败反弹 5/17）。v68 混合头：

- **box6 主干**（保留 v67 全部结构收益）：6 分布 = 平行四边形 (cx,cy,ux,uy,vx,vy)，
  对边平行由构造保证，v6 式角点各自塌缩结构性不可能；
- **角点残差头**（v68 新增）：4 角 × (dx,dy) = 8 分布，端点 = c±u±v + clamp(res, ±RES_LIM)
  ——残差补直回归精度（透视偏移/拟合损失由残差表达），限幅保塌缩防御
  （残差 8px@256 级，简并仍需 box 主干塌缩+残差拉满，结构性不可能不变）；
- **闸放宽**：残差头吸收非平行残差后，真实池平行四边形闸 3.5%→20%（剔 487→148 帧，
  真实监督 1285→1624 帧，残差统计 20% 闸内 99.9% 帧无损）。

接口契约：输入 [1,3,256,256] float CHW ImageNet 归一化（corner_affine 长边方形窗）；
输出按序 (simcc_x, simcc_y) 各 [1,7,512]——x 头 (cx,ux,vx,dxTL,dxTR,dxBL,dxBR)、
y 头 (cy,uy,vy,dyTL,dyTR,dyBL,dyBR)，域 BOX_LO/BOX_HI + [-RES_LIM,RES_LIM]。
架构主体不动（DW 骨干），proj 通道 6→14。算子全守 export.py 白名单。
"""
import torch
import torch.nn as nn

from cell_model import DWBlk

CORNER_IN = 256   # 输入边长（端上 POSE_INPUT）
NPK = 6           # box 参数数：cx, cy, ux, uy, vx, vy
NRES = 8          # 残差分量数：4 角 × (dx,dy)，序 [dxTL,dyTL,dxTR,dyTR,dxBL,dyBL,dxBR,dyBR]
NCH = NPK + NRES  # proj 通道总数 14
SIMCC_BINS = 512  # 每分布 bin 数
FEAT = 32         # 空间特征图边长
NH = 7            # 每头分布数（x 头 cx,ux,vx+4dx；y 头 cy,uy,vy+4dy）
X_CH = (0, 2, 4, 6, 8, 10, 12)   # x 头消费的 proj 通道（横向语义，行池化剖面）
Y_CH = (1, 3, 5, 7, 9, 11, 13)   # y 头消费的 proj 通道（纵向语义，列池化剖面）
RES_LIM = 8.0     # 残差限幅 px@256（伪标签残差统计：20% 相对残差闸内 99.9% 帧无损）

# box 参数值域（256 空间，512 bins 线性映射）。cx/cy 域即输入幅面；形状参数
# ±128 覆盖照片域满幅棋盘 + 旋转/剪切交叉项。残差域统一 [-RES_LIM, RES_LIM]。
BOX_LO = (0.0, 0.0, -128.0, -128.0, -128.0, -128.0)
BOX_HI = (256.0, 256.0, 128.0, 128.0, 128.0, 128.0)


def quad_to_gt(q):
    """格网四角 [TL,TR,BL,BR]（[...,4,2]，numpy）→ (box6 [...,6], res8 [...,8])。

    box6 = 平行四边形拟合（c=(TL+BR)/2, u=(TR-TL)/2, v=(BL-TL)/2）；
    res8 = 真四角 − box 重构端点（渲染源严格平行≈0；真实源=透视偏移，训练时
    由 corner_train clamp 到 ±RES_LIM——此处返回原值，截断在 target 侧做）。
    """
    import numpy as np
    tl, tr, bl, br = q[..., 0, :], q[..., 1, :], q[..., 2, :], q[..., 3, :]
    c = (tl + br) / 2.0
    u = (tr - tl) / 2.0
    v = (bl - tl) / 2.0
    box = np.concatenate([c, u, v], -1)
    rec = np.stack([c - u - v, c + u - v, c - u + v, c + u + v], -2)  # TL,TR,BL,BR
    res = (q - rec).reshape(*q.shape[:-2], 8)                          # 角优先 8 分量
    return box, res


def corners_from_parts(box, res8):
    """(box6, res8) → 四角 [TL,TR,BL,BR]（[...,4,2]），残差 clamp ±RES_LIM。

    torch 版（训练/val 用）；numpy 推理侧见 corner_infer.corners_from_parts。
    """
    cx, cy, ux, uy, vx, vy = [box[..., i] for i in range(6)]
    lim = RES_LIM
    r = [torch.clamp(res8[..., i], -lim, lim) for i in range(8)]
    corner = lambda px, py: torch.stack([px, py], -1)
    return torch.stack([
        corner(cx - ux - vx + r[0], cy - uy - vy + r[1]),   # TL
        corner(cx + ux - vx + r[2], cy + uy - vy + r[3]),   # TR
        corner(cx - ux + vx + r[4], cy - uy + vy + r[5]),   # BL
        corner(cx + ux + vx + r[6], cy + uy + vy + r[7])], -2)  # BR


class CornerSimCC(nn.Module):
    """256×256×3 → (head_x [B,7,512], head_y [B,7,512])（box6+残差双头）。

    proj 出 14 通道分布激活图（序 box6+res8），x 头消费 X_CH 通道的行池化剖面、
    y 头消费 Y_CH 通道的列池化剖面——横向语义走竖线证据、纵向走水平线证据。
    """

    def __init__(self, width=(48, 64, 96, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width[0]), nn.ReLU(inplace=True))
        strides = (2, 2, 1, 1, 1)  # 128→64→32→32→32→32（保 32×32 空间，剖面 32 bin）
        self.blocks = nn.Sequential(*[
            DWBlk(width[i], width[i + 1], strides[i]) for i in range(len(strides))])
        self.proj = nn.Conv2d(width[-1], NCH, 1, bias=False)      # [B,14,32,32]
        self.pool_x = nn.Conv2d(NCH, NCH, (FEAT, 1), stride=(FEAT, 1), groups=NCH, bias=False)
        self.pool_y = nn.Conv2d(NCH, NCH, (1, FEAT), stride=(1, FEAT), groups=NCH, bias=False)
        nn.init.constant_(self.pool_x.weight, 1.0 / FEAT)
        nn.init.constant_(self.pool_y.weight, 1.0 / FEAT)
        self.fc_x = nn.Linear(NH * FEAT, NH * SIMCC_BINS)
        self.fc_y = nn.Linear(NH * FEAT, NH * SIMCC_BINS)

    def forward(self, x):
        f = self.blocks(self.stem(x))          # [B,C,32,32]
        t = self.proj(f)                       # [B,14,32,32]
        gx = self.pool_x(t).flatten(2)         # [B,14,32]
        gy = self.pool_y(t).flatten(2)         # [B,14,32]
        b = x.shape[0]
        hx = self.fc_x(gx[:, list(X_CH)].reshape(b, -1)).view(b, NH, SIMCC_BINS)
        hy = self.fc_y(gy[:, list(Y_CH)].reshape(b, -1)).view(b, NH, SIMCC_BINS)
        return hx, hy


if __name__ == "__main__":
    import numpy as np
    m = CornerSimCC()
    n = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n/1e3:.1f}K（fp32 ≈ {n*4/1e6:.2f}MB）")
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, CORNER_IN, CORNER_IN))
    print("前向:", y[0].shape, y[1].shape)
    # quad → (box,res) → 端点 往返自测（真平行四边形：残差应≈0）
    q = torch.tensor([[[10., 20.], [110., 30.], [20., 220.], [120., 230.]]])
    box, res = quad_to_gt(q.numpy())
    q2 = corners_from_parts(torch.from_numpy(box), torch.from_numpy(res))
    print("往返 max|Δ|:", (q2 - q).abs().max().item(),
          "| 平行四边形残差 max:", np.abs(res).max())
