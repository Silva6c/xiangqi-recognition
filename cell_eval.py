# -*- coding: utf-8 -*-
"""cell_cls 格级评测（v63 第 1 波验收）。

26 帧金口径 warp 帧（315×280）逐格抠 crop（40px→128 上采样）独立推理，对账：
  1) 全盘 2072 判定格独立错误数（验收线 ≤8）
  2) r3 的 11 错格逐格重判表（验收线 ≥7 判对）
  3) 与 r3 错格重叠度（验收线 <50%）+ cell_cls 新增错清单（级联误伤风险预警）

注：warp 帧抠格为低清分布（31.5px 上采样）——原生高清格重抠（App 帧源图 quads）
另链补做；本脚本先锁低清下限能力。
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from cell_model import CellCls

CLS_CHARS = ".xKABNRCPkabnrcp"
ERR_JSON = r"D:\AI-works\cabinet\.cache\r3_11errors.json"


def crop_cells(img315: np.ndarray, labels: np.ndarray):
    """315×280 帧抠 90 格（格心公式与 model.py _cell_centers 同源），返回 [90,128,128,3] 与标签。"""
    out, labs = [], []
    for r in range(10):
        for c in range(9):
            cx, cy = 17.5 + 30.625 * c, 17.5 + 31.111 * r
            half = 20  # 40px crop（格距 31.5，含邻缘）
            x0, y0 = max(int(cx - half), 0), max(int(cy - half), 0)
            x1, y1 = min(int(cx + half), 280), min(int(cy + half), 315)
            crop = Image.fromarray(img315[y0:y1, x0:x1]).resize((128, 128), Image.LANCZOS)
            out.append(np.array(crop))
            labs.append(labels[r * 9 + c])
    return np.stack(out), np.array(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cell_v1/best.pt")
    ap.add_argument("--npz", default="data/eval_photo_human_v2.npz")
    ap.add_argument("--hd", action="store_true", help="用 eval_cells_hd.npz 原生高清格评测")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = CellCls().cuda() if torch.cuda.is_available() else CellCls()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    dev = next(model.parameters()).device

    if args.hd:
        d = np.load("data/eval_cells_hd.npz", allow_pickle=True)
        cells_all, labels = d["cells"], d["labels"]  # [26,90,128,128,3] 已抠好
    else:
        d = np.load(args.npz, allow_pickle=True)
        imgs, labels = d["imgs"], d["labels"]
    r3_errors = {(e["f"], tuple(e["cell"])): e for e in json.load(open(ERR_JSON, encoding="utf-8"))}

    errs, r3_fix, new_errs, preds_all = [], [], [], {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for f in range(cells_all.shape[0] if args.hd else imgs.shape[0]):
            if args.hd:
                cells, labs = cells_all[f], labels[f]
                x = torch.from_numpy(cells.astype(np.float32)).permute(0, 3, 1, 2).div(255).to(dev)
            else:
                cells, labs = crop_cells(imgs[f], labels[f])
                x = torch.from_numpy(cells).permute(0, 3, 1, 2).float().div(255).to(dev)
            pred = model(x).argmax(1).cpu().numpy()
            preds_all[f] = pred
            for i in range(90):
                if labs[i] == -100:  # IGNORE 不计分
                    continue
                r, c = divmod(i, 9)
                if pred[i] != labs[i]:
                    errs.append((f, r, c, CLS_CHARS[labs[i]], CLS_CHARS[pred[i]]))
                    if (f, (r, c)) not in r3_errors:
                        new_errs.append((f, r, c, CLS_CHARS[labs[i]], CLS_CHARS[pred[i]]))

    # r3 的 11 错格重判
    tbl = []
    for (f, cell), e in sorted(r3_errors.items()):
        p = preds_all[f][cell[0] * 9 + cell[1]]
        tbl.append({"f": f, "cell": list(cell), "true": e["true"], "r3_pred": e["pred"],
                    "cell_pred": CLS_CHARS[p], "翻正": bool(CLS_CHARS[p] == e["true"])})
    nfix = sum(t["翻正"] for t in tbl)
    overlap = 11 - nfix

    rep = {
        "ckpt": args.ckpt,
        "总错格": len(errs),
        "r3_11错格翻正": f"{nfix}/11",
        "错格重叠(r3错且cell错)": overlap,
        "重叠率": round(overlap / 11, 3),
        "cell新增错格数": len(new_errs),
        "11错格重判表": tbl,
        "全部错格": errs,
        "新增错格": new_errs,
    }
    out = args.out or args.ckpt.replace("best.pt", "eval_human26_cells.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("11错格重判表", "全部错格", "新增错格")},
                     ensure_ascii=False, indent=1))
    print("\n11 错格重判:")
    for t in tbl:
        print(f"  f{t['f']:02d}{tuple(t['cell'])} 真值{t['true']} r3:{t['r3_pred']} "
              f"cell_cls:{t['cell_pred']} {'✓翻正' if t['翻正'] else '✗'}")
    if new_errs:
        print(f"\ncell_cls 新增错格（级联误伤风险，{len(new_errs)} 个）:")
        for f, r, c, gt, p in new_errs[:20]:
            print(f"  f{f:02d}({r},{c}) 真值{gt} 预测{p}")
    print(f"\n已写出: {out}")


if __name__ == "__main__":
    main()
