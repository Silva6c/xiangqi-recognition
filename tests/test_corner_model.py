# -*- coding: utf-8 -*-
"""识别引擎最小回归测试（纯 CPU，秒级）——改模型/解码/几何后必跑。

运行：仓库根目录 `python -m pytest tests/ -q` 或 `python tests/test_corner_model.py`。
覆盖：模型前向契约（[B,7,512] 双头）、quad→(box6,res8)→四角往返、平行四边形
残差为零性、残差限幅、box 参数值域边界。对应协作手册剧本 C「单测必绿」。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from corner_model import (BOX_LO, BOX_HI, NPK, NRES, RES_LIM, SIMCC_BINS,
                          CornerSimCC, corners_from_parts, quad_to_gt)


def test_forward_shape():
    """前向契约：双头各 [B,7,512]（x 头 box3+残差4，y 头同构）。"""
    m = CornerSimCC().eval()
    with torch.no_grad():
        hx, hy = m(torch.randn(2, 3, 256, 256))
    assert hx.shape == (2, 7, SIMCC_BINS) and hy.shape == (2, 7, SIMCC_BINS)


def test_quad_to_gt_roundtrip_parallel():
    """真平行四边形：box 重构端点残差必须为 0（渲染源 GT 无损的根基）。"""
    q = np.array([[[10, 20], [110, 30], [20, 220], [120, 230]]], np.float32)
    box, res = quad_to_gt(q)
    assert np.abs(res).max() == 0.0


def test_quad_to_gt_perspective_residual():
    """非平行四边形（透视）：残差=真四角−box 端点，量级有限可被残差头表达。"""
    q = np.array([[[0, 0], [100, 10], [5, 100], [105, 95]]], np.float32)
    _, res = quad_to_gt(q)
    assert res.shape == (1, NRES)
    assert 0 < np.abs(res).max() < 20.0  # px@256，残差头 ±8 限幅域附近


def test_corners_from_parts_clamp():
    """残差限幅：超出 ±RES_LIM 的分量被截断（塌缩防御不被残差破坏）。"""
    box = torch.tensor([[128.0, 128.0, 44.0, 0.0, 0.0, 56.0]])
    res = torch.tensor([[99.0, -99.0, 0, 0, 0, 0, 0, 0]])  # 远超限幅
    c = corners_from_parts(box, res)
    assert abs(c[0, 0, 0].item() - (128 - 44 + RES_LIM)) < 1e-5   # TL.x = 84+8
    assert abs(c[0, 0, 1].item() - (128 - 56 - RES_LIM)) < 1e-5   # TL.y = 72-8


def test_box_domain_constants():
    """值域契约：cx/cy ∈ (0,256)，形状参数 ±128（与端上 boxDecode 硬编码逐项对齐）。"""
    assert BOX_LO == (0.0, 0.0, -128.0, -128.0, -128.0, -128.0)
    assert BOX_HI == (256.0, 256.0, 128.0, 128.0, 128.0, 128.0)
    assert RES_LIM == 8.0


def test_param_count_budget():
    """体积预算：混合头 fp32 ≤ 8MB（judges v68 注记的工程约束）。"""
    n = sum(p.numel() for p in CornerSimCC().parameters())
    assert n * 4 / 1e6 <= 8.0, f"fp32 {n*4/1e6:.2f}MB 超预算"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 项全绿")
