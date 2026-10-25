# -*- coding: utf-8 -*-
"""邻域分桶评测（v38 微观判据工具，docs/v38性能研究与渲染提速方案-20261017.md §四）。

用指定权重对 gen 四轴评测集做 CPU/前向推理，统计占子格错误率按 8 邻域有子数的
分桶曲线（U 型：孤立子=上下文真空、拥挤子=相互干扰）、漏检/幻觉率、行列分布。
v38a（注意力）vs v38b（备案）vs 基线的架构起效微观证据 = U 型两端向谷值收敛。

用法：
  python eval_ctx_bucket.py --weights runs/v38a/gen_best.pt --width 256,384,640,896 \
      --head-dim 256 --attn True [--axes style,piece,angle,roboflow] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import GridHead

IM_M = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_S = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
ROWS, COLS = 10, 9


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def neighbors(r: int, c: int) -> list:
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < ROWS and 0 <= c + dc < COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="torch ckpt（如 runs/v38a/gen_best.pt）")
    ap.add_argument("--width", default="256,384,640,896")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--attn", type=str2bool, default=False)
    ap.add_argument("--aux-head", type=str2bool, default=False,
                    help="ckpt 含 aux_fc 权重时须开（v38a 系 ckpt 默认带）")
    ap.add_argument("--axes", default="style,piece,angle,roboflow")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--json", default="", help="结果落盘路径（同时打印）")
    args = ap.parse_args()

    width = tuple(int(x) for x in args.width.split(","))
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = GridHead(width=width, head_dim=args.head_dim, attn=args.attn, aux_head=args.aux_head)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"加载 {args.weights}（step {ck.get('step')}，attn={args.attn}，CPU 前向）")

    bucket: dict[int, list[int]] = {}
    col_err = np.zeros((COLS, 2)); row_err = np.zeros((ROWS, 2))
    axis_stat: dict[str, dict] = {}
    miss_all = [0, 0]; hall_all = [0, 0]; occ_all = [0, 0]
    for axis in args.axes.split(","):
        p = Path(__file__).parent / f"data/eval_gen_{axis}.npz"
        if not p.exists():
            print(f"跳过 {axis}（{p.name} 不存在）")
            continue
        z = np.load(p)
        imgs, labs = z["imgs"], z["labels"]
        preds = []
        with torch.no_grad():
            for i in range(0, len(imgs), args.batch):
                x = torch.from_numpy(imgs[i:i + args.batch].transpose(0, 3, 1, 2).copy()).float().div_(255)
                x = (x - torch.from_numpy(IM_M)) / torch.from_numpy(IM_S)
                preds.append(model(x).argmax(2).numpy())
        preds = np.concatenate(preds)
        valid = labs != -100
        occ = (labs != 0) & valid
        empty = (labs == 0) & valid
        err = (preds != labs) & valid
        miss = int((err & occ).sum()); hall = int((err & empty).sum())
        axis_stat[axis] = {"miss": miss, "occ": int(occ.sum()), "hall": hall, "empty": int(empty.sum())}
        miss_all[0] += miss; miss_all[1] += int(occ.sum())
        hall_all[0] += hall; hall_all[1] += int(empty.sum())
        occ_all[0] += int((err & occ).sum()); occ_all[1] += int(occ.sum())
        for n in range(len(imgs)):
            lab = labs[n].reshape(ROWS, COLS)
            e = err[n].reshape(ROWS, COLS)
            for r in range(ROWS):
                for c in range(COLS):
                    if lab[r, c] in (-100, 0):
                        continue
                    k = sum(1 for rr, cc in neighbors(r, c) if lab[rr, cc] not in (-100, 0))
                    s = bucket.setdefault(k, [0, 0])
                    s[0] += int(e[r, c]); s[1] += 1
                    col_err[c] += [int(e[r, c]), 1]
                    row_err[r] += [int(e[r, c]), 1]

    result = {
        "weights": args.weights, "step": ck.get("step"),
        "axis": {a: {"miss_rate": round(v["miss"] / max(1, v["occ"]), 4),
                     "hall_rate": round(v["hall"] / max(1, v["empty"]), 4)}
                 for a, v in axis_stat.items()},
        "miss_rate": round(miss_all[0] / max(1, miss_all[1]), 4),
        "hall_rate": round(hall_all[0] / max(1, hall_all[1]), 4),
        "occ_err_rate": round(occ_all[0] / max(1, occ_all[1]), 4),
        "bucket": {str(k): {"err": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)}
                   for k, v in sorted(bucket.items())},
        "col_rate": [round(col_err[c][0] / max(1, col_err[c][1]), 4) for c in range(COLS)],
        "row_rate": [round(row_err[r][0] / max(1, row_err[r][1]), 4) for r in range(ROWS)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {args.json}")


if __name__ == "__main__":
    main()
