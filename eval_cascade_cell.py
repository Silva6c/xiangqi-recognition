# -*- coding: utf-8 -*-
"""v63 纯自研级联评测：r3_w085 主力 + cell_cls 格级精分类器仲裁（零 nano）。

与 eval_cascade.py（v59 nano 版）同框架，仲裁位替换：
  L1 级联   —— r3 逐格 pmax<thr 的格采纳 cell_cls 的 argmax（可选双门：cell 自身
               置信 ≥ cthr 才采纳，两模型都慌则保持 r3）
  子数规则  —— 四档里 nano 指认位换成 cell_cls 指认（apply_rule_fix 的 nano= 参数）
  对换      —— 不变（r3 内部证据）

口径：
  26 帧 v2 双分布——HD（eval_cells_hd.npz 原生高清格，cell_cls 主打）与
  LO（315×280 帧抠格上采样，兜底/bench 场景同分布）；四轴只跑 L1（v59 定案：
  渲染随机盘不守子数规则）；bench 低清格。

用法：
  python eval_cascade_cell.py --ckpt runs/cell_v1/best.pt [--thr 0.90] [--cthr 0.0]
      [--json out.json] [--skip-axes]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import eval_cascade as EC
from cell_eval import crop_cells
from cell_model import CellCls

HERE = Path(__file__).resolve().parent
CLASSES = EC.CLASSES


def infer_frame(sess_r3, model_cell, dev, img, cells_feeded):
    """r3 onnx 单帧 + cell_cls 单帧（格 crop [90,128,128,3]）。"""
    out = EC.infer_r3(sess_r3, img)
    x = torch.from_numpy(cells_feeded.astype(np.float32)).permute(0, 3, 1, 2).div(255).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        lo = model_cell(x).float()
    p = torch.softmax(lo, dim=1)
    conf, cp = p.max(1)
    out["cell"] = cp.cpu().numpy()
    out["cell_conf"] = conf.cpu().numpy()
    return out


def cascade_cell(out, labels, thr, cthr, conflict=None):
    """纯自研三件套（cell 版）。返回 (L1, L2, L3, 事件)。
    子数指认档同受 cthr 门约束；conflict=(r3_thr, cell_thr) 开启双高置信冲突复核——
    r3 ≥r3_thr 且 cell ≥cell_thr 且意见相左时采纳 cell（打 r3 高置信错：f05 0.993 型，
    nano 版级联的盲区——nano 无校准置信不可反向比较）。"""
    top1, pmax, second = out["top1"], out["pmax"], out["second"]
    cell, cconf = out["cell"], out["cell_conf"]

    def adopt(p):
        """gate 采纳 + 帅将保护。
        （帧级一致率闸曾试：26 帧翻正帧被误闸 2→5、共识帧仅 94→86 双输已回滚——
        低质帧风险防线放部署侧：级联改格数 >15% 判 bad_frame 走重试，见 v63 方案 §五。）"""
        gate = np.where(pmax < thr)[0]
        for g in gate:
            # 帅将保护：r3 判 K/k 的格不覆盖——r3 全盘错型从无 K/k（26 帧史证），
            # cell 的 K/k 判别是软肋（v3 实测 f06(0,4)/f14 两误伤皆此型）
            if cconf[g] >= cthr and top1[g] not in (2, 9):
                p[g] = cell[g]
        return gate

    def conflict_adopt(p):
        if conflict is None:
            return []
        r3t, ct = conflict
        # 帅将保护：r3 或 cell 一方判 K/k 的冲突不自动采纳（v3 实测 f06(0,4) k→n /
        # f14 K→. 误伤——cell 的 K/k 判别是软肋，每盘唯一子的硬约束交给子数规则）
        hits = np.where((pmax >= r3t) & (cconf >= ct) & (cell != p)
                        & ~np.isin(p, (2, 9)) & ~np.isin(cell, (2, 9)))[0]
        for g in hits:
            p[g] = cell[g]
        return [int(g) for g in hits]

    p1 = top1.copy()
    adopt(p1)
    conflict_adopt(p1)
    if int((p1 != 0).sum()) >= 8:   # 空盘/残局帧不跑子数（占子<8 时超编多为幻视，
        p2, _ = apply_rule_fix_cell(p1.copy(), second, pmax, cell, cconf, cthr)  # 翻格连锁伤空格——共识帧教训）
    else:
        p2 = p1.copy()
    swaps = EC.find_swaps(top1, second, pmax, thr)
    locked = set()
    p3 = top1.copy()
    for g1, g2 in swaps:
        p3[g1], p3[g2] = p3[g2], p3[g1]
        locked |= {g1, g2}
    for g in np.where(pmax < thr)[0]:
        if g not in locked and cconf[g] >= cthr and top1[g] not in (2, 9):
            p3[g] = cell[g]
    cf3 = conflict_adopt(p3)
    if int((p3 != 0).sum()) >= 8:
        p3, ev3 = apply_rule_fix_cell(p3, second, pmax, cell, cconf, cthr)
    else:
        ev3 = []
    ev = {"gate": [int(g) for g in np.where(pmax < thr)[0]],
          "adopt": [int(g) for g in np.where(pmax < thr)[0] if cconf[g] >= cthr],
          "conflict": cf3,
          "rule": ev3,
          "swap": [{"cells": [(g1 // 9, g1 % 9), (g2 // 9, g2 % 9)]} for g1, g2 in swaps]}
    return (EC.err_count(p1, labels), EC.err_count(p2, labels), EC.err_count(p3, labels), ev)


def apply_rule_fix_cell(pred, second, pmax, cell, cconf=None, cthr=0.0):
    """子数四档（cell 版）：① cell 指认 ② r3 次选 ③ cell 空格 ④ 次选空格。
    ①③ 档要求 cell_conf ≥ cthr（低置信 cell 无指认资格——v1 实测无门时误伤）。"""
    events = []
    for _ in range(10):
        counts = {c: int((pred == c).sum()) for c in EC.LIMITS}
        overs = [(c, counts[c] - l) for c, l in EC.LIMITS.items() if counts[c] > l]
        if not overs:
            break
        progressed = False
        for cls, _over in overs:
            if cls in (2, 9):
                continue  # 帅将类不参与子数翻格——cell 的 K/k 指认不可信（f06(0,4) k→n 误翻教训），
                          # 且帅将位置错=盘面级错，子数证据修不动
            deficit = {c for c, l in EC.LIMITS.items() if counts[c] < l}
            cand = []
            for g in np.where(pred == cls)[0]:
                s2, cp = int(second[g]), int(cell[g])
                cc = 1.0 if cconf is None else float(cconf[g])
                if cp in deficit and cc >= cthr:
                    cand.append((0, float(pmax[g]), g, cp))
                elif s2 in deficit:
                    cand.append((1, float(pmax[g]), g, s2))
                elif cp == 0 and cc >= cthr:
                    cand.append((2, float(pmax[g]), g, 0))
                elif s2 == 0:
                    cand.append((3, float(pmax[g]), g, 0))
            if not cand:
                continue
            cand.sort()
            _, _, g, new = cand[0]
            events.append({"cell": (int(g // 9), int(g % 9)), "old": CLASSES[cls],
                           "new": CLASSES[new], "src": "cell指认" if new == int(cell[g]) else "次选",
                           "p": round(float(pmax[g]), 3)})
            pred[g] = new
            progressed = True
        if not progressed:
            break
    return pred, events


def eval_26(name, imgs, labels, hd_cells, sess_r3, model_cell, dev, thr, cthr, conflict=None):
    """26 帧双分布：HD 与 LO 各跑一遍三件套。"""
    res = {}
    for tag, feeder in (("HD", hd_cells), ("LO", None)):
        l0 = l1 = l2 = l3 = 0
        t0 = time.time()
        detail = []
        for k in range(len(imgs)):
            cells = hd_cells[k] if feeder is not None else crop_cells(imgs[k], labels[k])[0]
            out = infer_frame(sess_r3, model_cell, dev, imgs[k], cells)
            l0 += EC.err_count(out["top1"], labels[k])
            a1, a2, a3, ev = cascade_cell(out, labels[k], thr, cthr, conflict)
            l1 += a1; l2 += a2; l3 += a3
            if ev["adopt"] or ev["rule"] or ev["swap"] or ev["conflict"]:
                detail.append({"f": k, **{kk: vv for kk, vv in ev.items()}})
        res[f"{name}_{tag}"] = {"L0_r3": l0, "L1_级联": l1, "L2_级联+子数": l2,
                                "L3_完整": l3, "sec": round(time.time() - t0, 1),
                                "事件": detail}
        print(f"[{name}-{tag}] L0={l0} → L1={l1} → L2={l2} → L3={l3}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cell_v1/best.pt")
    ap.add_argument("--thr", type=float, default=0.90)
    ap.add_argument("--cthr", type=float, default=0.0)
    ap.add_argument("--conflict", default="", help="r3阈值,cell阈值（如 0.97,0.98）开双高冲突复核")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-axes", action="store_true")
    ap.add_argument("--r3-onnx", default="",
                    help="覆盖默认 r3 fp32（部署口径对账用：传端上 int8 onnx 测级联真实水平）")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellCls().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    sess_r3 = ort.InferenceSession(args.r3_onnx or EC.R3, providers=["CPUExecutionProvider"])

    conflict = tuple(float(x) for x in args.conflict.split(",")) if args.conflict else None
    report = {"thr": args.thr, "cthr": args.cthr, "conflict": conflict,
              "ckpt": args.ckpt, "r3": args.r3_onnx or EC.R3, "sets": {}}

    z = np.load(str(HERE / "data/eval_photo_human_v2.npz"), allow_pickle=False)
    imgs, labs = z["imgs"], z["labels"]
    hd = np.load(str(HERE / "data/eval_cells_hd.npz"), allow_pickle=True)["cells"]
    report["sets"].update(eval_26("human26v2", imgs, labs, hd, sess_r3, model, dev,
                                  args.thr, args.cthr, conflict))

    z = np.load(str(HERE / "data/eval_bench.npz"), allow_pickle=False)
    l0 = l1 = l2 = l3 = 0
    for k in range(len(z["imgs"])):
        cells = crop_cells(z["imgs"][k], z["labels"][k])[0]
        out = infer_frame(sess_r3, model, dev, z["imgs"][k], cells)
        l0 += EC.err_count(out["top1"], z["labels"][k])
        a1, a2, a3, _ = cascade_cell(out, z["labels"][k], args.thr, args.cthr, conflict)
        l1 += a1; l2 += a2; l3 += a3
    report["sets"]["bench10_LO"] = {"L0_r3": l0, "L1": l1, "L2": l2, "L3": l3}
    print(f"[bench10-LO] L0={l0} → L1={l1} → L2={l2} → L3={l3}")

    if not args.skip_axes:
        for axis in ("style", "piece", "angle", "roboflow"):
            p = HERE / "data" / f"eval_gen_{axis}.npz"
            if not p.exists():
                continue
            za = np.load(str(p), allow_pickle=False)
            n = len(za["imgs"])
            l0 = l1 = 0
            occ_n = occ_hit0 = occ_hit1 = 0
            t0 = time.time()
            for k in range(n):
                cells = crop_cells(za["imgs"][k], za["labels"][k])[0]
                out = infer_frame(sess_r3, model, dev, za["imgs"][k], cells)
                p1 = out["top1"].copy()
                for g in np.where(out["pmax"] < args.thr)[0]:
                    if out["cell_conf"][g] >= args.cthr:
                        p1[g] = out["cell"][g]
                lab = za["labels"][k]
                l0 += EC.err_count(out["top1"], lab)
                l1 += EC.err_count(p1, lab)
                m = (lab != -100) & (lab != 0)      # 占子格掩码
                occ_n += int(m.sum())
                occ_hit0 += int(((out["top1"] == lab) & m).sum())
                occ_hit1 += int(((p1 == lab) & m).sum())
            row = {"L0_r3_err": l0, "L1_级联_err": l1,
                   "r3_occ": round(occ_hit0 / occ_n, 4),
                   "L1_occ": round(occ_hit1 / occ_n, 4)}
            print(f"[gen_{axis}] L0 err={l0}(occ {row['r3_occ']}) → L1 err={l1}(occ {row['L1_occ']}) "
                  f"({time.time()-t0:.0f}s)")
            report["sets"][f"gen_{axis}"] = row

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"已存 {args.json}")


if __name__ == "__main__":
    main()
