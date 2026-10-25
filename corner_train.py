# -*- coding: utf-8 -*-
"""自研角点模型（CornerSimCC）轻量训练（v64）。cell_cls 级轻训练授权范围内（1~2h/轮）。

用法：python corner_train.py --run-name corner_v1 --steps 4000 --batch 32
损失：SimCC 一维高斯软标签 soft-CE（σ bins 可扫参；负样本 target=均匀分布、权重 0.5）。
val：固定 seed 渲整屏帧，指标 = 256 空间帧平均角点误差 px（越小越好，best 按此保存）+
ok@6px 率（6px@256 ≈ 25px 原图）。
纪律：cudnn.benchmark 永禁；单重进程（跑前确认无其他 python 训练残留）。
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from corner_data import CornerFeeder, make_val
from corner_model import CORNER_IN, NKP, SIMCC_BINS, CornerSimCC

torch.backends.cudnn.benchmark = False


def simcc_target(pts, valid, sigma, device):
    """角点 [B,4,2]（256 空间 px）→ 归一化高斯软标签 (tx, ty) [B,4,512]。

    负样本（valid=0）→ 均匀分布（教压平峰值，端上质量门拦截）。
    """
    ar = torch.arange(SIMCC_BINS, device=device)
    cx = (pts[..., 0] * 2.0).clamp(1.0, SIMCC_BINS - 2.0)   # 像素 → bin（×2）；防御 clamp
    cy = (pts[..., 1] * 2.0).clamp(1.0, SIMCC_BINS - 2.0)
    tx = torch.exp(-0.5 * ((ar[None, None] - cx[:, :, None]) / sigma) ** 2)
    ty = torch.exp(-0.5 * ((ar[None, None] - cy[:, :, None]) / sigma) ** 2)
    tx = tx / tx.sum(-1, keepdim=True)
    ty = ty / ty.sum(-1, keepdim=True)
    uni = torch.full_like(tx, 1.0 / SIMCC_BINS)
    keep = (valid > 0)[:, None, None]
    return torch.where(keep, tx, uni), torch.where(keep, ty, uni)


def decode_px(lx, ly):
    """SimCC logits → 256 空间角点像素坐标 [B,4,2]（端上 argmax/512×256 同构）。"""
    px = lx.argmax(-1) / SIMCC_BINS * CORNER_IN
    py = ly.argmax(-1) / SIMCC_BINS * CORNER_IN
    return torch.stack([px, py], dim=-1)


@torch.no_grad()
def validate(model, vx, vpts):
    model.eval()
    lx, ly = model(vx)
    pred = decode_px(lx, ly)
    d = torch.linalg.norm(pred - vpts, dim=-1).mean(-1)   # [B] 帧平均角点误差
    model.train()
    return d.mean().item(), (d < 6.0).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="corner_v1")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--sigma", type=float, default=2.5, help="SimCC 软标签 σ（bin 单位）")
    ap.add_argument("--seed", type=int, default=6400)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=48)
    ap.add_argument("--neg-ratio", type=float, default=0.08)
    ap.add_argument("--real-npz", default=None, help="伪标签池（第 1 波）")
    ap.add_argument("--real-ratio", type=float, default=0.30)
    args = ap.parse_args()

    out = Path(f"runs/{args.run_name}")
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = open(out / "log.txt", "a", encoding="utf-8")

    def P(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    torch.manual_seed(args.seed)
    model = CornerSimCC().to(dev)
    P(f"[corner_train] 参数 {sum(p.numel() for p in model.parameters())/1e3:.1f}K "
      f"σ={args.sigma} 设备 {dev}")

    # v6 提速：workers 4→8（渲染预取喂饱 GPU——batch 96 时 4 线程喂不满是 42 分钟瓶颈）
    feeder = CornerFeeder(args.seed, args.batch, neg_ratio=args.neg_ratio,
                          real_npz=args.real_npz, real_ratio=args.real_ratio, workers=8)
    vx, vpts = make_val(777, args.val_n)
    vx, vpts = vx.to(dev), vpts.to(dev)
    P(f"[val 集] {args.val_n} 帧固定 seed 渲染")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    best = 1e9
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, pts, valid = feeder.next()
        x = x.to(dev, non_blocking=True)
        pts, valid = pts.to(dev), valid.to(dev)
        tx, ty = simcc_target(pts, valid, args.sigma, dev)
        w = torch.where(valid > 0, torch.ones_like(valid), 0.5 * torch.ones_like(valid))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            lx, ly = model(x)
            lp_x, lp_y = F.log_softmax(lx.float(), dim=-1), F.log_softmax(ly.float(), dim=-1)
            loss = ((-(tx * lp_x).sum(-1)).mean(-1) * w).mean() + \
                   ((-(ty * lp_y).sum(-1)).mean(-1) * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if step % 50 == 0:
            P(f"step {step} loss {loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} "
              f"elapsed {(time.time()-t0)/60:.1f}min")
        if step % args.val_every == 0 or step == args.steps:
            mpx, ok6 = validate(model, vx, vpts)
            P(f"[val] step {step} mean_px {mpx:.2f} ok@6px {ok6:.3f}")
            if mpx < best:
                best = mpx
                torch.save(model.state_dict(), out / "best.pt")
                P(f"[save] best mean_px {mpx:.2f}")
    torch.save(model.state_dict(), out / "final.pt")
    feeder.close()
    json.dump({"best_val_mean_px": best, "steps": args.steps, "sigma": args.sigma,
               "params_k": sum(p.numel() for p in model.parameters()) / 1e3},
              open(out / "summary.json", "w"))
    P(f"[done] best mean_px {best:.2f}px 总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
