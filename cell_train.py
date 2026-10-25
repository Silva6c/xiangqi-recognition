# -*- coding: utf-8 -*-
"""cell_cls 轻量训练（v63 第 1 波）。批量授权范围内的 10~20 分钟级轻训练。

用法：python cell_train.py --run-name cell_v1 --steps 12000
纪律：cudnn.benchmark 永禁；单重进程（跑前确认无其他 python 训练残留）。
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cell_data import CellFeeder, render_board_cells, augment
from cell_model import CellCls

torch.backends.cudnn.benchmark = False


def make_val(seed: int, n: int, fonts):
    """固定 seed 渲染 val 集（与训练同增广链、seed 固定 → 可复现）。"""
    import random
    rng = random.Random(seed)
    imgs, labs = [], []
    while len(imgs) < n:
        for pil, lab in render_board_cells(rng, fonts):
            if len(imgs) >= n:
                break
            imgs.append(augment(pil, rng))
            labs.append(lab)
    import numpy as np
    return (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255),
            torch.tensor(labs, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="cell_v1")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2100)
    ap.add_argument("--val-every", type=int, default=300)
    ap.add_argument("--val-n", type=int, default=768)
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
    model = CellCls().to(dev)
    P(f"[cell_train] 参数 {sum(p.numel() for p in model.parameters())/1e3:.1f}K 设备 {dev}")

    feeder = CellFeeder(args.seed, args.batch)
    val_fonts = feeder.fonts
    vx, vy = make_val(999, args.val_n, val_fonts)
    vx, vy = vx.to(dev), vy.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    best = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = feeder.next()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if step % 100 == 0:
            P(f"step {step} loss {loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} "
              f"elapsed {(time.time()-t0)/60:.1f}min")
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = model(vx).argmax(1)
            acc = (pred == vy).float().mean().item()
            model.train()
            P(f"[val] step {step} acc {acc:.4f}")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), out / "best.pt")
                P(f"[save] best {acc:.4f}")
    torch.save(model.state_dict(), out / "final.pt")
    feeder.close()
    json.dump({"best_val_acc": best, "steps": args.steps,
               "params_k": sum(p.numel() for p in model.parameters()) / 1e3},
              open(out / "summary.json", "w"))
    P(f"[done] best_val {best:.4f} 总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
