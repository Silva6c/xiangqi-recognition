# -*- coding: utf-8 -*-
"""级联系统评测（v59 系统超越轮）：r3_w085 主力 + 三件套后处理，对拍 nano。

管线（最终口径 L3 = 对换 → 级联 → 子数，对换在原始预测上检测）：
  对换   —— 同行相邻两格原始预测为同方 {N,R}（或 {n,r}）且 r3 logits 互为次选
            （g1 的 top2==g2 的 pred 且 g2 的 top2==g1 的 pred）且至少一格 p<thr
            ——交叉证据+低置信双门，防真 N/R 相邻双高置信误换
  级联   —— r3 逐格 softmax 置信 p<thr 且未被对换决定的格，采纳 nano 的 argmax
            （单向门控：只用 r3 的置信触发，nano 全局低置信不可反向比较）
  子数   —— 每方每类上限（帅1/士象马车炮2/兵5）超编时，翻"次选指向缺编类"的
            超编格到其次选（次选不指向缺编类则宁可不翻——防盲翻真子）

分级输出：L0 r3 基线 / L1 仅级联 / L2 级联+子数 / L3 完整三件套。

附带两问：
  蒸馏平反 —— nano batch=1 重测渲染域（四轴 npz 即现渲染器帧）占子格 argmax 判对率
              （当年判死刑的 0.0 系 batch=64 BN 批污染疑似假象）
  循环对账 —— 26 帧按「nano 预测==真值」（造卷循环成分高）与 !=（人工独立性强）
              分子集对拍 r3/nano/级联错格

用法:
  python eval_cascade.py [--thr 0.90] [--json 输出.json] [--skip-axes 只跑26帧+bench]
                         [--r3 指定onnx路径]   # v60 训后专项重放用（默认 r3_w085）
  python eval_cascade.py --tta [--json 输出.json]   # v60 §1.1 TTA 自仲裁（纯自研，无 nano）
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
R3 = str(HERE / "../../../.cache/v56_study/r3_hbw08_w085/gridhead_fp32.onnx")
NANO = str(HERE / "../../app/src/main/assets/models/nano_v3-0319.onnx")
IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
CLASSES = ".xKABNRCPkabnrcp"
SPLIT = 17
# 每类上限: 索引→数量（红 2-8, 黑 9-15）
LIMITS = {2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 5,
          9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 5}
SWAP_PAIRS = ({5, 6}, {12, 13})          # 同方 N/R、n/r 对换家族


def infer_pair(sess_r3, sess_nano, img):
    """单帧 batch=1 双模型推理。返回 dict（top1/pmax/second/nano 各 [90]）。"""
    x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = np.ascontiguousarray(((x - IM_MEAN) / IM_STD)[None])
    lo = sess_r3.run(None, {"input": x})[0][0]          # [90,16] logits
    order = np.argsort(-lo, axis=1)
    top1, second = order[:, 0], order[:, 1]
    e = np.exp(lo - lo.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    pmax = p[np.arange(90), top1]
    nlo = sess_nano.run(None, {sess_nano.get_inputs()[0].name: x})[0][0]
    return {"top1": top1, "pmax": pmax, "second": second, "nano": nlo.argmax(axis=1)}


# ── TTA 自仲裁模式（v60 §1.1）：纯自研系统级底数，不依赖 nano ────────────
TTA_VIEWS = ((0.98, 0.0, 0.0), (1.02, 0.0, 0.0),    # 中心缩放（角点内缩/外扩）
             (1.0, 2.0, 0.0), (1.0, -2.0, 0.0))     # 平移 ±2px（水平，角点误差主分量）


def infer_r3(sess_r3, img):
    """单帧 r3 推理。返回 dict（top1/pmax/second 各 [90]）。"""
    x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = np.ascontiguousarray(((x - IM_MEAN) / IM_STD)[None])
    lo = sess_r3.run(None, {"input": x})[0][0]
    order = np.argsort(-lo, axis=1)
    top1, second = order[:, 0], order[:, 1]
    e = np.exp(lo - lo.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return {"top1": top1, "pmax": p[np.arange(90), top1], "second": second}


def infer_r3_tta(sess_r3, img):
    """TTA 多视图推理：原图+4 扰动视图逐格 logits 平均 → 新 top1/pmax/second。

    扰动是"角点误差等效型"（warp 摆正残差的缩放/平移近似），非当年已证伪的
    输入级 ±4px 平移（网格错位型）——幅度减半且混合缩放，历史警示见 v60 方案。
    """
    def logits(im):
        x = im.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = np.ascontiguousarray(((x - IM_MEAN) / IM_STD)[None])
        return sess_r3.run(None, {"input": x})[0][0]
    acc = logits(img)
    H, W = img.shape[:2]
    for s, tx, ty in TTA_VIEWS:
        M = np.float32([[s, 0.0, (1 - s) * W / 2 + tx], [0.0, s, (1 - s) * H / 2 + ty]])
        v = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        acc = acc + logits(v)
    acc /= 1 + len(TTA_VIEWS)
    order = np.argsort(-acc, axis=1)
    top1, second = order[:, 0], order[:, 1]
    e = np.exp(acc - acc.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return {"top1": top1, "pmax": p[np.arange(90), top1], "second": second}


def eval_tta(name, imgs, labels, sess_r3):
    """纯自研 TTA 自仲裁评测：T0 TTA平均 / T1 TTA+子数规则（nano=None 分支）。

    逐格对账输出修复格（原错→对）与新增错格（原对→错），判定口径=新增错 ≤1。
    """
    n = len(imgs)
    l0 = t0 = t1 = s1 = 0
    fixed, newbad = [], []
    t0b = time.time()
    for k in range(n):
        base = infer_r3(sess_r3, imgs[k])
        tta = infer_r3_tta(sess_r3, imgs[k])
        valid = labels[k] != -100
        l0 += err_count(base["top1"], labels[k])
        t0 += err_count(tta["top1"], labels[k])
        # 单帧+子数（nano=None 纯自研分支）= 纯自研系统底数
        pb, _evb = apply_rule_fix(base["top1"].copy(), base["second"], base["pmax"], nano=None)
        s1 += err_count(pb, labels[k])
        b_ok = base["top1"] == labels[k]
        t_ok = tta["top1"] == labels[k]
        for g in np.where(valid & b_ok & ~t_ok)[0]:
            newbad.append({"f": k, "cell": (int(g // 9), int(g % 9)),
                           "old": CLASSES[int(base["top1"][g])],
                           "new": CLASSES[int(tta["top1"][g])]})
        for g in np.where(valid & ~b_ok & t_ok)[0]:
            fixed.append({"f": k, "cell": (int(g // 9), int(g % 9)),
                          "old": CLASSES[int(base["top1"][g])],
                          "new": CLASSES[int(tta["top1"][g])]})
        p1, _ev = apply_rule_fix(tta["top1"].copy(), tta["second"], tta["pmax"], nano=None)
        t1 += err_count(p1, labels[k])
    row = {"L0_单帧": l0, "S1_单帧+子数": s1, "T0_TTA平均": t0, "T1_TTA+子数": t1,
           "修复格数": len(fixed), "新增错格数": len(newbad),
           "修复明细": fixed, "新增错明细": newbad,
           "sec": round(time.time() - t0b, 1)}
    print(f"[{name}] L0 单帧={l0} / S1 单帧+子数={s1} → T0 TTA平均={t0} → T1 TTA+子数={t1}  "
          f"(修复 {len(fixed)} / 新增错 {len(newbad)}, {row['sec']}s)")
    return row


def find_swaps(top1, second, pmax, thr):
    """原始预测上找对换对：同行相邻同方 {N,R} + 互为次选 + 至少一格 p<thr。"""
    pairs = []
    for r in range(10):
        for c in range(8):
            g1, g2 = r * 9 + c, r * 9 + c + 1
            p1, p2 = top1[g1], top1[g2]
            if {p1, p2} not in SWAP_PAIRS or p1 == p2:
                continue
            if second[g1] == p2 and second[g2] == p1 and (pmax[g1] < thr or pmax[g2] < thr):
                pairs.append((g1, g2))
    return pairs


def apply_rule_fix(pred, second, pmax, nano=None):
    """子数超编全局一致性仲裁（迭代修复，动态重算计数）。

    每轮对每个超编类选一个候选格翻类，四档证据优先级：
      ① nano 预测∈缺编类（nano 指认）  ② r3 次选∈缺编类
      ③ nano 预测=空格（幻视去子）      ④ r3 次选=空格
    同档按 r3 置信最低优先；无候选则该类放弃（宁留超编不盲翻）。
    """
    events = []
    for _ in range(10):                                   # 迭代修复上限
        counts = {c: int((pred == c).sum()) for c in LIMITS}
        overs = [(c, counts[c] - l) for c, l in LIMITS.items() if counts[c] > l]
        if not overs:
            break
        progressed = False
        for cls, _over in overs:                          # 每轮每类最多翻一格
            deficit = {c for c, l in LIMITS.items() if counts[c] < l}
            cand = []
            for g in np.where(pred == cls)[0]:
                s2 = int(second[g])
                npd = int(nano[g]) if nano is not None else -1
                if npd in deficit:
                    cand.append((0, float(pmax[g]), g, npd))
                elif s2 in deficit:
                    cand.append((1, float(pmax[g]), g, s2))
                elif nano is not None and npd == 0:
                    cand.append((2, float(pmax[g]), g, 0))
                elif s2 == 0:
                    cand.append((3, float(pmax[g]), g, 0))
            if not cand:
                continue
            cand.sort()
            _, _, g, new = cand[0]
            src = "nano指认" if new == (int(nano[g]) if nano is not None else -1) else "次选"
            events.append({"cell": (int(g // 9), int(g % 9)),
                           "old": CLASSES[cls], "new": CLASSES[new], "src": src,
                           "p": round(float(pmax[g]), 3)})
            pred[g] = new
            progressed = True
        if not progressed:
            break
    return pred, events


def err_count(pred, labels):
    valid = labels != -100
    return int(((pred != labels) & valid).sum())


def cascade_frame(out, labels, thr):
    """单帧三件套。返回 (L1仅级联, L2级联+子数, L3完整, 事件dict)。"""
    top1, pmax, second, nano = out["top1"], out["pmax"], out["second"], out["nano"]
    # L1: 仅级联
    p1 = top1.copy()
    for g in np.where(pmax < thr)[0]:
        p1[g] = nano[g]
    # L2: 级联+子数
    p2, ev_rule = apply_rule_fix(p1.copy(), second, pmax, nano)
    # L3: 完整（对换在原始预测上检测，对换格不参与级联）
    swaps = find_swaps(top1, second, pmax, thr)
    locked = set()
    p3 = top1.copy()
    for g1, g2 in swaps:
        p3[g1], p3[g2] = p3[g2], p3[g1]
        locked |= {g1, g2}
    for g in np.where(pmax < thr)[0]:
        if g not in locked:
            p3[g] = nano[g]
    p3, ev_rule3 = apply_rule_fix(p3, second, pmax, nano)
    ev = {"gate": [int(g) for g in np.where(pmax < thr)[0]],
          "rule": ev_rule, "rule_l3": ev_rule3,
          "swap": [{"cells": [(g1 // 9, g1 % 9), (g2 // 9, g2 % 9)]} for g1, g2 in swaps]}
    return (err_count(p1, labels), err_count(p2, labels), err_count(p3, labels), ev)


def eval_set(name, imgs, labels, sess_r3, sess_nano, thr, detail=False):
    """跑一套评测集。返回 (row, r3_pred[n,90], nano_pred[n,90])。"""
    n = len(imgs)
    l0 = l1 = l2 = l3 = 0
    events = {"gate": [], "rule": [], "swap": []}
    r3_preds = np.zeros((n, 90), np.int64)
    nano_preds = np.zeros((n, 90), np.int64)
    t0 = time.time()
    for k in range(n):
        out = infer_pair(sess_r3, sess_nano, imgs[k])
        r3_preds[k], nano_preds[k] = out["top1"], out["nano"]
        l0 += err_count(out["top1"], labels[k])
        a1, a2, a3, ev = cascade_frame(out, labels[k], thr)
        l1 += a1; l2 += a2; l3 += a3
        if detail:
            events["gate"] += [{"f": k, "g": g} for g in ev["gate"]]
            events["rule"] += [{"f": k, **e} for e in ev["rule_l3"]]
            events["swap"] += [{"f": k, **e} for e in ev["swap"]]
    row = {"L0_r3": l0, "L1_级联": l1, "L2_级联+子数": l2, "L3_完整三件套": l3,
           "sec": round(time.time() - t0, 1)}
    print(f"[{name}] L0 r3基线={l0} → L1 仅级联={l1} → L2 级联+子数={l2} "
          f"→ L3 完整三件套={l3}  ({row['sec']}s)")
    if detail:
        row.update(events)
    return row, r3_preds, nano_preds


def main():
    thr, out_json, skip_axes, tta = 0.90, "", False, False
    r3_path = ""
    a = sys.argv[1:]
    while "--thr" in a:
        i = a.index("--thr"); thr = float(a[i + 1]); a = a[:i] + a[i + 2:]
    while "--json" in a:
        i = a.index("--json"); out_json = a[i + 1]; a = a[:i] + a[i + 2:]
    while "--r3" in a:
        i = a.index("--r3"); r3_path = a[i + 1]; a = a[:i] + a[i + 2:]
    if "--skip-axes" in a:
        skip_axes = True
        i = a.index("--skip-axes"); a = a[:i] + a[i + 1:]
    if "--tta" in a:
        tta = True
        i = a.index("--tta"); a = a[:i] + a[i + 1:]

    if r3_path:
        global R3
        R3 = r3_path

    # ── TTA 自仲裁模式（v60 §1.1）：纯自研系统底数，只跑 26 帧+bench ──
    if tta:
        sess_r3 = ort.InferenceSession(R3, providers=["CPUExecutionProvider"])
        report = {"mode": "tta", "views": 1 + len(TTA_VIEWS), "sets": {}}
        z = np.load(str(HERE / "data/eval_photo_human_v2.npz"), allow_pickle=False)
        report["sets"]["human26v2"] = eval_tta("human26v2", z["imgs"], z["labels"], sess_r3)
        z = np.load(str(HERE / "data/eval_bench.npz"), allow_pickle=False)
        report["sets"]["bench10"] = eval_tta("bench10", z["imgs"], z["labels"], sess_r3)
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
            print(f"已存 {out_json}")
        return

    sess_r3 = ort.InferenceSession(R3, providers=["CPUExecutionProvider"])
    sess_nano = ort.InferenceSession(NANO, providers=["CPUExecutionProvider"])
    report = {"thr": thr, "r3": "r3_w085 fp32", "sets": {}}

    # ── 26 帧 v2（逐格明细 + 分域 + 循环对账）────────────
    z = np.load(str(HERE / "data/eval_photo_human_v2.npz"), allow_pickle=False)
    imgs, labs = z["imgs"], z["labels"]
    row26, r3p, nap = eval_set("human26v2", imgs, labs, sess_r3, sess_nano, thr, detail=True)
    # 分域重放（逐帧级联三件套）
    for nm, sl in (("实物17", slice(0, 17)), ("App9", slice(17, 26))):
        e = sum(cascade_frame(infer_pair(sess_r3, sess_nano, imgs[k]), labs[k], thr)[2]
                for k in range(sl.start, sl.stop))
        row26[nm + "_L3"] = int(e)
        print(f"    {nm}: 完整三件套错格={int(e)}")
    valid = labs != -100
    nano_ok = (nap == labs) & valid
    nano_bad = (nap != labs) & valid
    row26["对账"] = {
        "nano对格数": int(nano_ok.sum()), "nano错格数": int(nano_bad.sum()),
        "r3错在nano对格": int(((r3p != labs) & nano_ok).sum()),
        "r3错在nano错格": int(((r3p != labs) & nano_bad).sum())}
    print(f"    循环对账: nano 对 {row26['对账']['nano对格数']} 格"
          f"(其中 r3 错 {row26['对账']['r3错在nano对格']}) / "
          f"nano 错 {row26['对账']['nano错格数']} 格"
          f"(其中 r3 错 {row26['对账']['r3错在nano错格']})")
    report["sets"]["human26v2"] = row26

    # ── bench 10 帧 ──────────────────────────────────────
    z = np.load(str(HERE / "data/eval_bench.npz"), allow_pickle=False)
    rowb, _, _ = eval_set("bench10", z["imgs"], z["labels"], sess_r3, sess_nano, thr)
    report["sets"]["bench10"] = rowb

    # ── 四轴（渲染域）：级联防退 + 蒸馏平反同批 ────────────
    if not skip_axes:
        for axis in ("style", "piece", "angle", "roboflow"):
            p = HERE / "data" / f"eval_gen_{axis}.npz"
            if not p.exists():
                print(f"[跳过] {axis} 轴缺失")
                continue
            za = np.load(str(p), allow_pickle=False)
            rowa, r3pa, napa = eval_set(f"gen_{axis}", za["imgs"], za["labels"],
                                        sess_r3, sess_nano, thr)
            la = za["labels"]
            va = la != -100
            occ = va & (la != 0)
            hit = float(((napa == la) & occ).sum() / occ.sum())    # 蒸馏平反口径
            cell = float(((napa == la) & va).sum() / va.sum())
            r3occ = float(((r3pa == la) & occ).sum() / occ.sum())
            r3cell = float(((r3pa == la) & va).sum() / va.sum())
            rowa.update({"nano_占子格判对率": round(hit, 4), "nano_cell": round(cell, 4),
                         "r3_occ": round(r3occ, 4), "r3_cell": round(r3cell, 4)})
            # 级联后 occ（L1 口径重算：低置信格采纳 nano）
            print(f"    {axis}: r3 occ={r3occ:.4f} | nano 占子格判对率={hit:.4f} "
                  f"cell={cell:.4f}（当年死刑口径=0.0）")
            report["sets"][f"gen_{axis}"] = rowa

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"已存 {out_json}")


if __name__ == "__main__":
    main()
