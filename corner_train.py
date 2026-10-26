# -*- coding: utf-8 -*-
"""自研角点模型（v68 混合头）轻量训练。cell_cls 级轻训练授权范围内（1~2h/轮）。

用法：python corner_train.py --run-name corner_v68 --steps 1500 --batch 96 --lr 3e-3 \
        --real-npz data/pools/corner_v5_pool.npz --val-every 100
损失：14 分布 soft-CE = box6（cx,cy,ux,uy,vx,vy）+ 残差8（4 角 dx,dy）；x 头
（cx,ux,vx+4dx）7 分布、y 头（cy,uy,vy+4dy）7 分布。值域 BOX_LO/HI + ±RES_LIM
（残差 target clamp 限幅——真实源透视残差超限部分截断，统计 99.9% 帧无损）。
负样本 target=均匀分布、权重 0.5。
val：固定 seed 渲整屏帧，端点（c±u±v+clamp 残差）四角误差 px@256 + ok@6px。
纪律：cudnn.benchmark 永禁；单重进程。
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from corner_data import CornerFeeder, make_val
from corner_model import (BOX_LO, BOX_HI, RES_LIM, CORNER_IN, NPK, NRES, SIMCC_BINS,
                          X_CH, Y_CH, CornerSimCC, corners_from_parts)

torch.backends.cudnn.benchmark = False

# 14 维全局参数布局：box6 [0..5] + res8 [6..13]；值域 box 用 BOX_LO/HI、res ±RES_LIM
_LO = torch.tensor(list(BOX_LO) + [-RES_LIM] * NRES)
_HI = torch.tensor(list(BOX_HI) + [RES_LIM] * NRES)
_X_IDX = list(X_CH)   # x 头分布对应的全局维（cx,ux,vx,dxTL,dxTR,dxBL,dxBR）
_Y_IDX = list(Y_CH)   # y 头（cy,uy,vy,dyTL,dyTR,dyBL,dyBR）
NALL = NPK + NRES


def gt_target(gt, valid, sigma, device):
    """gt [B,14]（box6+res8，256 空间）→ (t_x, t_y) 各 [B,7,512] 高斯软标签。

    res 分量 clamp ±RES_LIM（真实源透视残差超限截断）；负样本（valid=0）→ 均匀。
    """
    lo = _LO.to(device)[None]
    hi = _HI.to(device)[None]
    center = ((gt - lo) / (hi - lo) * SIMCC_BINS).clamp(1.0, SIMCC_BINS - 2.0)
    ar = torch.arange(SIMCC_BINS, device=device)
    t = torch.exp(-0.5 * ((ar[None, None, :] - center[:, :, None]) / sigma) ** 2)
    t = t / t.sum(-1, keepdim=True)
    uni = torch.full_like(t, 1.0 / SIMCC_BINS)
    keep = (valid > 0)[:, None, None]
    t = torch.where(keep, t, uni)
    return t[:, _X_IDX], t[:, _Y_IDX]


def decode_gt(hx, hy):
    """(hx, hy) [B,7,512] → 14 维参数 [B,14]（box6+res8，端上 argmax+域换算同构）。"""
    b = hx.shape[0]
    out = hx.new_zeros(b, NALL)
    for j, p in enumerate(_X_IDX):
        out[:, p] = hx[:, j].argmax(-1).float() / SIMCC_BINS * (_HI[p] - _LO[p]) + _LO[p]
    for j, p in enumerate(_Y_IDX):
        out[:, p] = hy[:, j].argmax(-1).float() / SIMCC_BINS * (_HI[p] - _LO[p]) + _LO[p]
    return out


@torch.no_grad()
def validate(model, vx, vgt):
    model.eval()
    hx, hy = model(vx)
    gt = decode_gt(hx, hy)
    corners = corners_from_parts(gt[:, :NPK], gt[:, NPK:])       # [B,4,2] 端点（含 clamp 残差）
    corners_ref = corners_from_parts(vgt[:, :NPK], vgt[:, NPK:])  # GT 参照（残差也 clamp，公平口径）
    d = torch.linalg.norm(corners - corners_ref, dim=-1).mean(-1)
    model.train()
    return d.mean().item(), (d < 6.0).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="corner_v68")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sigma", type=float, default=2.5, help="SimCC 软标签 σ（bin 单位）")
    ap.add_argument("--seed", type=int, default=6800)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=48)
    ap.add_argument("--neg-ratio", type=float, default=0.08)
    ap.add_argument("--real-npz", default=None, help="伪标签池（逗号分隔可多池）")
    ap.add_argument("--real-ratio", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=16,
                    help="渲染预取线程数（拉满档：本机 20 核 → 16，渲染吞吐是瓶颈主项）")
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
    P(f"[corner_train v68 混合头] 参数 {sum(p.numel() for p in model.parameters())/1e3:.1f}K "
      f"σ={args.sigma} RES_LIM={RES_LIM} 设备 {dev}")

    feeder = CornerFeeder(args.seed, args.batch, neg_ratio=args.neg_ratio,
                          real_npz=args.real_npz, real_ratio=args.real_ratio,
                          workers=args.workers)
    if feeder.pool is not None:
        P(f"[真实池] {len(feeder.pool)} 帧（平行四边形闸剔除 {feeder.pool.n_dropped}）")
    vx, vgt = make_val(777, args.val_n)
    vx, vgt = vx.to(dev), vgt.to(dev)
    P(f"[val 集] {args.val_n} 帧固定 seed 渲染")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    best = 1e9
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, gt, valid = feeder.next()
        x = x.to(dev, non_blocking=True)
        gt, valid = gt.to(dev), valid.to(dev)
        tx, ty = gt_target(gt, valid, args.sigma, dev)
        w = torch.where(valid > 0, torch.ones_like(valid), 0.5 * torch.ones_like(valid))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            hx, hy = model(x)
            lp_x, lp_y = F.log_softmax(hx.float(), dim=-1), F.log_softmax(hy.float(), dim=-1)
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
            mpx, ok6 = validate(model, vx, vgt)
            P(f"[val] step {step} mean_px {mpx:.2f} ok@6px {ok6:.3f}")
            if mpx < best:
                best = mpx
                torch.save(model.state_dict(), out / "best.pt")
                P(f"[save] best mean_px {mpx:.2f}")
    torch.save(model.state_dict(), out / "final.pt")
    feeder.close()
    json.dump({"best_val_mean_px": best, "steps": args.steps, "sigma": args.sigma,
               "params_k": sum(p.numel() for p in model.parameters()) / 1e3,
               "param": "box6_plus_residual8_hybrid"},
              open(out / "summary.json", "w"))
    P(f"[done] best mean_px {best:.2f}px 总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
