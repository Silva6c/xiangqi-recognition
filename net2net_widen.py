# -*- coding: utf-8 -*-
"""Net2Net 函数保持扩宽（GridHead 132K → ~288K，v20 容量扩宽主攻）。

背景：v18/v19 证明新渲染先验与实物域在 132K 容量内结构性冲突（域跷跷板），
且 v18 在自己的训练分布上 occ 仅 0.53（nano 0.82）——同分布欠拟合，容量是头号嫌疑。
v15 的"从头加宽"证伪不适用：本脚本从训练态 ckpt 出发做**严格函数保持**的宽度扩展
（Net2Net wide 版本：输出通道复制 + BN 同步复制 + 下游输入权重置零），
扩宽后初始 logits 与源模型全等（脚本内自验证），训练起点无损。

扩宽规则（逐层传播链，forward 逐位等价）：
- 输出通道扩宽的层（stem.0 / s*.pw / head.0）：新增输出通道参数复制自随机源通道；
  紧随的 BN（weight/bias/running_mean/running_var）同步复制；
- depthwise（dw）通道与输入一一对应：随上游输出扩宽直接复制
  （其输出同位复制，由下游 pw 的输入置零兜住）；
- 消费扩宽通道的普通 Conv/Linear（s*.pw 输入侧、head.0 输入侧、fc 输入侧）：
  新增输入通道权重**置零** → 输出严格不变（全部 Conv 无 bias，零输入贡献为零）；
- 新增通道链路从零权重起步，但梯度链路完整（零权重的梯度 = 激活 × 下游梯度 ≠ 0），
  训练中正常生长；源通道下游权重非零、新通道为零，无对称死区。

用法：
  python net2net_widen.py --src runs/v19/ckpt.pt --out runs/v20/net2net_init.pt \
      --width 96,144,240,336 --head-dim 96
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import GridHead, count_params


def widen_out(t: torch.Tensor, n_new: int) -> torch.Tensor:
    """[C,...] 沿第 0 维扩宽：新增通道复制自随机源通道"""
    if n_new <= 0:
        return t
    idx = torch.randint(0, t.shape[0], (n_new,))
    return torch.cat([t, t[idx].clone()], 0)


def widen_in_zero(t: torch.Tensor, n_new: int) -> torch.Tensor:
    """[O,I,...] 沿第 1 维扩宽：新增输入通道权重置零（无 bias，严格函数保持）"""
    if n_new <= 0:
        return t
    z = torch.zeros(*t.shape[:1], n_new, *t.shape[2:], dtype=t.dtype)
    return torch.cat([t, z], 1)


def widen_bn(sd: dict, key: str, n_new: int) -> None:
    """BN 全部仿射参数与 running stats 沿通道维复制扩宽"""
    if n_new <= 0:
        return
    for k in ("weight", "bias", "running_mean", "running_var"):
        sd[f"{key}.{k}"] = widen_out(sd[f"{key}.{k}"], n_new)
    # num_batches_tracked 是标量计数器，保留原值


def widen_sdblock(sd: dict, name: str, d_in: tuple[int, int], d_out: tuple[int, int]) -> None:
    """扩宽一个 DWBlock（name∈s2/s3/s4.0/s4.1）：dw/bn1 随输入扩，pw 输入置零+输出复制，bn2 随输出扩"""
    old_in, new_in = d_in
    old_out, new_out = d_out
    if new_in > old_in:   # depthwise 通道数 = 输入通道数 = 输出通道数
        sd[f"{name}.dw.weight"] = widen_out(sd[f"{name}.dw.weight"], new_in - old_in)
        widen_bn(sd, f"{name}.bn1", new_in - old_in)
    pw = sd[f"{name}.pw.weight"]
    pw = widen_in_zero(pw, new_in - old_in)          # 输入侧：消费扩宽通道，置零
    sd[f"{name}.pw.weight"] = widen_out(pw, new_out - old_out)   # 输出侧：复制
    widen_bn(sd, f"{name}.bn2", new_out - old_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源 ckpt（训练态权重，如 runs/v19/ckpt.pt）")
    ap.add_argument("--out", required=True, help="输出 ckpt 路径（供 train.py --init-from 使用）")
    ap.add_argument("--width", default="96,144,240,336", help="目标宽度四元组")
    ap.add_argument("--head-dim", type=int, default=96, help="目标头部压缩通道数")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    src_ck = torch.load(args.src, map_location="cpu", weights_only=False)
    src_sd = {k: v.clone() for k, v in src_ck["model"].items()}
    # 源宽度从权重形状自动推断（v35 起：支持任意世代 ckpt——v2~v19 为 132K，
    # v33b/v34 为 (192,288,480,672)+192 的 1.128M，不再硬编码）
    w1 = src_sd["stem.0.weight"].shape[0]
    w2 = src_sd["s2.pw.weight"].shape[0]
    w3 = src_sd["s3.pw.weight"].shape[0]
    w4 = src_sd["s4.1.pw.weight"].shape[0]
    D = src_sd["head.0.weight"].shape[0]
    src_width, src_head = (w1, w2, w3, w4), D
    print(f"源宽度（形状推断）: {src_width} head={src_head}")
    dst_width = tuple(int(x) for x in args.width.split(","))
    dst_head = args.head_dim
    assert len(dst_width) == 4
    assert all(n >= o for o, n in zip(src_width, dst_width)) and dst_head >= src_head, \
        "目标宽度必须 ≥ 源宽度（本脚本只扩不缩）"

    sd = src_sd
    dw1, dw2, dw3, dw4 = (n - o for o, n in zip(src_width, dst_width))
    dD = dst_head - src_head

    # stem：输出扩宽（3 通道输入不变）
    sd["stem.0.weight"] = widen_out(sd["stem.0.weight"], dw1)
    widen_bn(sd, "stem.1", dw1)
    # s2/s3/s4.0/s4.1：逐块扩宽
    widen_sdblock(sd, "s2", (w1, w1 + dw1), (w2, w2 + dw2))
    widen_sdblock(sd, "s3", (w2, w2 + dw2), (w3, w3 + dw3))
    widen_sdblock(sd, "s4.0", (w3, w3 + dw3), (w4, w4 + dw4))
    widen_sdblock(sd, "s4.1", (w4, w4 + dw4), (w4, w4 + dw4))
    # head：输入随 w4 扩（置零），输出随 head_dim 扩（复制）
    sd["head.0.weight"] = widen_out(widen_in_zero(sd["head.0.weight"], dw4), dD)
    widen_bn(sd, "head.1", dD)
    # fc：输入随 head_dim 扩（列置零），输出 16 类不变
    sd["fc.weight"] = widen_in_zero(sd["fc.weight"], dD)

    # ---- 函数保持自验证：扩宽前后同批输入 logits 必须全等 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_src = GridHead(width=src_width, head_dim=src_head).to(device).eval()
    m_dst = GridHead(width=dst_width, head_dim=dst_head).to(device).eval()
    m_src.load_state_dict(src_ck["model"])
    m_dst.load_state_dict(sd)
    # 一半随机输入 + 一半接近真实分布的有符号输入（覆盖各激活区间）
    torch.manual_seed(7)
    x = torch.randn(4, 3, 315, 280, device=device) * 2
    x = torch.cat([x, x.clamp(-2.2, 2.2)])
    with torch.no_grad():
        y_src, y_dst = m_src(x), m_dst(x)
    diff = float((y_src - y_dst).abs().max())
    print(f"函数保持验证：max|logits diff| = {diff:.2e}（应 < 1e-4）")
    assert diff < 1e-4, "扩宽后函数不等价——传播链有 bug，禁止用于训练"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in sd.items()}, "step": src_ck.get("step"),
                "src": args.src, "width": list(dst_width), "head_dim": dst_head,
                "note": "Net2Net 函数保持扩宽（net2net_widen.py），初始 logits 与源全等"}, out)
    print(f"参数量 {src_ck and count_params(m_src)/1e3:.1f}K → {count_params(m_dst)/1e3:.1f}K")
    print(f"落盘 {out}（--init-from 起点就绪）")


if __name__ == "__main__":
    main()
