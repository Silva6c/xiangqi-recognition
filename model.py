# -*- coding: utf-8 -*-
"""grid-head 轻量 CNN（自研识别引擎一期模型，docs/自研识别引擎方针.md 阶段2）。

架构（约 25 万参数，算子白名单：Conv/DepthwiseConv/ReLU/BatchNorm/Gemm）：
- 输入 [N,3,315,280]（端上 CLS_IN_H×CLS_IN_W 契约，ImageNet 归一化在数据侧完成）
- 首层内部 pad 到 320×288 使全部下采样整除（stride 8 → 特征图 40×36）
- backbone：stem(s2) + 三个 depthwise 阶段(s2/s2/s1)，通道 32→48→80→112
- 头部：1×1 conv 压到 40 维 → 「常数矩阵 GEMM」在 90 个格点中心做双线性采样
  （等价 grid_sample，但只依赖 MatMul+常量初始化器，规避端上 ORT-android 的算子兼容风险）
- 输出 [N,90,16] logits，行主序 cell=i/9 列=i%9，与 OnnxGeom.decodeCells 契约一致

「不确定就闭嘴」哲学的落点：本模块只出 logits；置信度（softmax 间距/温度）与
整盘弃权规则在 eval.py / 端上解码层实现，训练时以 ignore_index 处理遮挡格。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与渲染器/端上一致的几何常量
CLS_IN_H, CLS_IN_W = 315, 280
PADDED_H, PADDED_W = 320, 288     # 整除 stride 8
STRIDE = 8
FEAT_H, FEAT_W = PADDED_H // STRIDE, PADDED_W // STRIDE  # 40 × 36
ROWS, COLS, NCELLS = 10, 9, 90
NCLASS = 16

def _cell_centers() -> torch.Tensor:
    """90 个格点在 padded 输入坐标系 (H=320,W=288) 的中心像素坐标 [90,2] (x,y)。
    原始输入 280×315 中：cx = 17.5 + 30.625c，cy = 17.5 + 31.111r（来自
    网格角 (50,50)-(400,450) warp 空间经裁剪(25,25)与×0.7 缩放）；pad 偏移 (dx=4, dy=2)。"""
    xs = 17.5 + (400.0 - 100.0) * 0.7 / (COLS - 1) * torch.arange(COLS, dtype=torch.float64) + (PADDED_W - CLS_IN_W) / 2
    ys = 17.5 + (450.0 - 100.0) * 0.7 / (ROWS - 1) * torch.arange(ROWS, dtype=torch.float64) + (PADDED_H - CLS_IN_H) / 2
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [ROWS, COLS]
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [90,2] x,y

def _sample_matrix() -> torch.Tensor:
    """常数采样矩阵 M [90, CH*FEAT_H*FEAT_W]：第 i 行 = 在格点 i 的双线性 4 邻域权重
    （按通道复制）。输出 = flat @ M.T，仅依赖 MatMul。"""
    centers = _cell_centers() / STRIDE  # 特征图坐标
    M = torch.zeros(NCELLS, NCLASS * 0 + 1, FEAT_H, FEAT_W)  # 占位形状 [90,1,40,36]
    for i, (fx, fy) in enumerate(centers):
        x0, y0 = int(torch.floor(fx)), int(torch.floor(fy))
        dx, dy = float(fx - x0), float(fy - y0)
        for (xx, yy, w) in ((x0, y0, (1 - dx) * (1 - dy)), (x0 + 1, y0, dx * (1 - dy)),
                            (x0, y0 + 1, (1 - dx) * dy), (x0 + 1, y0 + 1, dx * dy)):
            M[i, 0, min(yy, FEAT_H - 1), min(xx, FEAT_W - 1)] += w
    return M.reshape(NCELLS, -1)  # [90, 40*36]

class DWBlock(nn.Module):
    """Depthwise Separable 块：DWConv3×3 → BN → ReLU → PWConv1×1 → BN → ReLU"""
    def __init__(self, ch_in: int, ch_out: int, stride: int):
        super().__init__()
        self.dw = nn.Conv2d(ch_in, ch_in, 3, stride=stride, padding=1, groups=ch_in, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.pw = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x

class GridHead(nn.Module):
    """输入 [N,3,315,280] → 输出 [N,90,16] logits"""

    def __init__(self, width=(64, 96, 160, 224), head_dim: int = 64,
                 attn: bool = False, aux_head: bool = False, xattn: bool = False):
        super().__init__()
        w1, w2, w3, w4 = width
        self.head_dim = head_dim
        # stem：pad 后 stride2。手写 pad 用 F.pad（常量 0），保持算子纯净
        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(w1), nn.ReLU())
        self.s2 = DWBlock(w1, w2, stride=2)
        self.s3 = DWBlock(w2, w3, stride=2)
        self.s4 = nn.Sequential(DWBlock(w3, w4, stride=1), DWBlock(w4, w4, stride=1))
        self.head = nn.Sequential(nn.Conv2d(w4, head_dim, 1, bias=False),
                                  nn.BatchNorm2d(head_dim), nn.ReLU())
        # 常数采样矩阵（buffer：随模型导出为 ONNX 初始化器，推理零成本）
        self.register_buffer("sample_m", _sample_matrix(), persistent=False)
        self.fc = nn.Linear(head_dim, NCLASS)
        # 格点 → 最近特征格的展平索引（用于把采样问题化为 GEMM：见 forward 注释）
        # 实现：head 输出 [N,head_dim,40,36] → reshape [N, head_dim*40*36]
        # logits = flat @ (M_onehot_per_channel).T  —— M 已按 [90, 40*36] 构建，
        # 通道维在 head 上（head_dim 与 40*36 分离），故用 einsum 等价的两次 reshape+matmul：
        # flat: [N, D, FH*FW] → 逐点 [N, D] @ fc 权重需先采样。这里用 M 对 FH*FW 维采样：
        # sampled[n, i, d] = Σ_{p} M[i, p] * flat[n, d, p]
        #   = (M @ flat_p.T).T → torch: torch.einsum('ip,ndp->nid', M, flat)
        # einsum 导出为 MatMul（Transpose + MatMul），白名单内。

        # ===== v38a 格间注意力（docs/v38性能研究与渲染提速方案-20261017.md §四）=====
        # 错误形态实证：邻域密度 U 型（孤立子 47.4% 错误率=上下文真空，邻格全空时局部
        # 机制无信息可卷，只有全局聚合能供证据）+ 漏检主导（4~5 倍于幻觉）。
        # 本块在 90 个格 token 间做内容自适应聚合，每格判决可参考其余 89 格。
        # 算子纪律：ReZero 残差（t + α·F(t)，α 初值 0=恒等映射，init-from 微调零扰动
        # 起步）替代 LayerNorm，FFN 用 ReLU 替代 GELU——新增算子仅 Softmax 一个。
        self.use_attn = attn
        self.use_aux = aux_head
        self.n_heads = 4                   # v38a/v62 注意力共用头数
        if attn:
            # QKV 拆三个独立 Linear（共享 Linear + 索引拆分会导出成 Gather，不在白名单；
            # 独立 Linear 直接落 MatMul，参数量相同）
            self.wq = nn.Linear(head_dim, head_dim, bias=True)
            self.wk = nn.Linear(head_dim, head_dim, bias=True)
            self.wv = nn.Linear(head_dim, head_dim, bias=True)
            self.proj = nn.Linear(head_dim, head_dim, bias=True)
            self.ffn = nn.Sequential(nn.Linear(head_dim, head_dim * 4, bias=True), nn.ReLU(),
                                     nn.Linear(head_dim * 4, head_dim, bias=True))
            # 90 格可学习位置嵌入：注意力置换不变性需要空间先验（棋盘 9×10 固定排列）
            self.pos_emb = nn.Parameter(torch.zeros(1, ROWS * COLS, head_dim))
            self.a1 = nn.Parameter(torch.zeros(1))   # ReZero α：注意力分支
            self.a2 = nn.Parameter(torch.zeros(1))   # ReZero α：FFN 分支
        if aux_head:
            # 占子辅助头（90 格 2 类）：训练期强迫特征对"有子/空"线性可分（漏检主导的
            # 直击手段）；导出/推理可剪，或作端上格级弃权信号
            self.aux_fc = nn.Linear(head_dim, 2)

        # ===== v62 cross-attention 解码层（docs/v62方案-90query解码层手术 §二）=====
        # 机制：常数矩阵 sample_m 是固定坐标采样——warp 角点误差/异形时采样点偏移
        # （TTA 实验实锤的几何敏感架构属性）。本块让 90 格 token 以可学习 query 对
        # 全特征图（40×36=1440 位置）做 cross-attention"搜索证据"，跨域鲁棒。
        # 位置：采样 token 之后、v38a 自注意力之前（取证据增强 → 格间推理）。
        # 算子纪律同 v38a：独立 Linear 落 MatMul + Softmax，端上白名单零新增。
        # ReZero α=β=0 起步 → init-from 后输出与源权重逐位一致（零扰动，血统保住）。
        self.use_xattn = xattn
        if xattn:
            self.xk = nn.Linear(head_dim, head_dim, bias=True)    # K 投影（特征图侧）
            self.xv = nn.Linear(head_dim, head_dim, bias=True)    # V 投影（特征图侧）
            self.xq = nn.Linear(head_dim, head_dim, bias=True)    # Q 投影（格 token 侧）
            self.xproj = nn.Linear(head_dim, head_dim, bias=True)
            self.xffn = nn.Sequential(nn.Linear(head_dim, head_dim * 4, bias=True), nn.ReLU(),
                                      nn.Linear(head_dim * 4, head_dim, bias=True))
            # K/V 位置编码：行+列嵌入分解 (FH+FW)×D（≈19.5K）替代完整 FH×FW×D（369K）
            self.kv_pos_r = nn.Parameter(torch.zeros(1, FEAT_H, 1, head_dim))
            self.kv_pos_c = nn.Parameter(torch.zeros(1, 1, FEAT_W, head_dim))
            self.xa1 = nn.Parameter(torch.zeros(1))   # ReZero α：cross-attn 分支
            self.xa2 = nn.Parameter(torch.zeros(1))   # ReZero β：FFN 分支

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        n = x.shape[0]
        x = F.pad(x, (4, 4, 2, 3))  # W:280→288, H:315→320（左右/上下）
        x = self.stem(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)                       # [N, w4, 40, 36]
        x = self.head(x)                     # [N, head_dim, 40, 36]
        feat2d = x.permute(0, 2, 3, 1)       # [N, FH, FW, D]（K/V 源，xattn 用）
        x = x.reshape(n, x.shape[1], FEAT_H * FEAT_W)   # [N, D, P]
        # 常数矩阵采样：[90,P] @ [N,P,D]（广播）→ [N,90,D]，纯 MatMul+Transpose
        x = torch.matmul(self.sample_m, x.transpose(1, 2))   # [N,90,D]
        if self.use_xattn:
            # v62 cross-attention：Q=90 格 token，K/V=全特征图（+行列分解位置编码）
            # α=0 起步（ReZero）——载入旧权重后本块输出恒为零增量，行为与源一致
            t0 = x
            h = self.n_heads
            dh = self.head_dim // h
            kv = feat2d + self.kv_pos_r + self.kv_pos_c        # [N,FH,FW,D] 广播相加
            kv = kv.reshape(n, FEAT_H * FEAT_W, self.head_dim)  # [N,P,D]
            q = self.xq(t0).reshape(n, ROWS * COLS, h, dh).permute(0, 2, 1, 3)
            k = self.xk(kv).reshape(n, FEAT_H * FEAT_W, h, dh).permute(0, 2, 1, 3)
            v = self.xv(kv).reshape(n, FEAT_H * FEAT_W, h, dh).permute(0, 2, 1, 3)
            att = torch.matmul(q, k.transpose(-2, -1)) * dh ** -0.5
            att = att.softmax(dim=-1)                          # [N,H,90,P]
            ctx = torch.matmul(att, v).transpose(1, 2).reshape(n, ROWS * COLS, self.head_dim)
            x = t0 + self.xa1 * self.xproj(ctx)                # ReZero 残差
            x = x + self.xa2 * self.xffn(x)
        if self.use_attn:
            t = x + self.pos_emb
            dh = self.head_dim // self.n_heads
            h = self.n_heads
            # [N,90,D] → [N,H,90,dh]（Reshape+Transpose，无 Gather）
            q = self.wq(t).reshape(n, ROWS * COLS, h, dh).permute(0, 2, 1, 3)
            k = self.wk(t).reshape(n, ROWS * COLS, h, dh).permute(0, 2, 1, 3)
            v = self.wv(t).reshape(n, ROWS * COLS, h, dh).permute(0, 2, 1, 3)
            att = torch.matmul(q, k.transpose(-2, -1)) * dh ** -0.5
            att = att.softmax(dim=-1)                       # [N,H,90,90]（新增算子：Softmax）
            ctx = torch.matmul(att, v).transpose(1, 2).reshape(n, ROWS * COLS, self.head_dim)
            t = t + self.a1 * self.proj(ctx)                # ReZero 残差
            t = t + self.a2 * self.ffn(t)
            x = t
        aux = self.aux_fc(x) if (return_aux and self.use_aux) else None
        logits = self.fc(x)                  # [N,90,16]
        if return_aux and self.use_aux:
            return logits, aux
        return logits

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    for kw in ({}, {"xattn": True, "attn": True, "aux_head": True}):
        m = GridHead(head_dim=256, width=(256, 384, 640, 896), **kw)
        m.eval()
        tag = "xattn" if kw.get("xattn") else "base"
        with torch.no_grad():
            y = m(torch.randn(2, 3, CLS_IN_H, CLS_IN_W))
        print(f"[{tag}] 参数 {count_params(m)/1e6:.2f}M 输出 {tuple(y.shape)}（期望 (2, 90, 16)）")
        if kw.get("xattn"):
            assert float(m.xa1.abs()) == 0.0 and float(m.xa2.abs()) == 0.0, "ReZero 初值必须为 0"
            print("[xattn] ReZero α/β 初值 0 ✓（零扰动起步）")
    # 采样矩阵完整性自检：每行权重和应为 1（双线性插值系数归一性）
    s = m.sample_m.sum(dim=1)
    print(f"采样矩阵行和 min={s.min():.6f} max={s.max():.6f}（应均为 1）")
    # 格点中心范围自检：应落在特征图内部
    c = _cell_centers() / STRIDE
    print(f"格点特征坐标 x∈[{c[:,0].min():.2f},{c[:,0].max():.2f}] y∈[{c[:,1].min():.2f},{c[:,1].max():.2f}]"
          f"（特征图 {FEAT_W}×{FEAT_H}）")
