# -*- coding: utf-8 -*-
"""v44 VN 原图裁片管线（docs/v44部署修复与实物扩量方案-20261017.md §2.2 扩展）。

背景：VN 三集（abplc/fpt/robot-pi，33695 框）大都是近景/特写图，整盘不可见——
v44 parse 实测只入池 +111 帧（角点失败 +403、结构不合法 +919）。但贴片路线的核心
论点正是「贴片只依赖框内图像+类别（COCO 保证），不需要格位」——绕过整帧管线
（角点对齐/结构校验），直接从原图裁 bbox 解锁全部框。

QC 语义（与 v43 原帧自证的区别与等价性）：
 VN 原图没有格位（无角点），无法在"来源格位"上自证。改为**开局标准位换子自证**：
 把裁片贴到该类开局标准位（替换渲染子——格位先验="有子"，与贴片类一致，无
 先验冲突；v43 渲染帧 QC 全灭的根因是"红仕贴黑区"式先验错配，此处按类匹配位），
 nano 单帧推理（batch=1！BN 批污染坑，v43 §6.1）双门（argmax==COCO 类 +
 conf≥0.65）——测的是「贴片图像本身 nano 可辨认且类正确」，坏片（裁废/叠压/
 标错）在库端全拦。QC 通过的片入库后自由贴到任何空位（打破先验是训练的事，
 QC 只管可辨认性）。

流程：
 1. 遍历 VN 三集 COCO json → 每框裁原图 bbox（外扩 25%，clip 边界）→ resize 38×38；
 2. 类别经 VN_MAPS 条件映射（abplc/fpt 正向 vua=王/tuong=象；robot-pi 反向
    tuong=将/tuongx=象——坐标分布+框数比核验）；
 3. 渲染底帧 → 占子格按类匹配贴候选片（K/k 位保留渲染帅将防双帅）→
    nano 一次推理验全格 → 合格片入库（一帧验 ~30 片，3.3 万框 ≈ 1100 帧推理）；
 4. 产出 data/vn_raw_patch_lib.npz（cls_idx→[N,38,38,3] u8），供
    augment_roboflow_patch.py --extra-lib 合并进贴片库。

用法：
  python augment_vn_raw_patch.py --limit-frames 40   # 冒烟（≈1200 片过 QC）
  python augment_vn_raw_patch.py                     # 全量 3.3 万框
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import augment_real_transplant as AT
import parse_roboflow_coco as P
import renderer as R
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
ROOT = HERE.parent.parent.parent / "data/roboflow_raw/detection_14class"
OUT_NPZ = HERE / "data/vn_raw_patch_lib.npz"
BAN = ("K", "k")                    # 防双帅：与主库同规则

CLS_CHARS = P.CLS_CHARS
IDX = {ch: i for i, ch in enumerate(CLS_CHARS)}
IDX_DOT = IDX["."]


def crop_boxes() -> tuple[dict[int, deque], Counter]:
    """遍历 VN 三集 COCO json，裁全部候选框 → 按类队列。返回 (queues, stat)。"""
    queues: dict[int, deque] = defaultdict(deque)
    stat = Counter()
    for proj, vnmap in P.VN_MAPS.items():
        for jp in sorted(ROOT.glob(f"{proj}/**/_annotations.coco.json")):
            split_dir = jp.parent
            d = json.load(open(jp, encoding="utf-8"))
            cats = {c["id"]: c["name"] for c in d.get("categories", [])}
            cat2cls = {cid: IDX[vnmap[n]] for cid, n in cats.items() if n in vnmap}
            id2file = {im["id"]: im["file_name"] for im in d.get("images", [])}
            anns = defaultdict(list)
            for a in d.get("annotations", []):
                if a["category_id"] in cat2cls:
                    anns[a["image_id"]].append(a)
            for img_id, fname in id2file.items():
                fpath = split_dir / fname
                img = cv2.imread(str(fpath))
                if img is None:
                    stat["读取失败"] += 1
                    continue
                h, w = img.shape[:2]
                for a in anns.get(img_id, []):
                    bx, by, bw, bh = a["bbox"]
                    if bw < 8 or bh < 8:
                        stat["框过小"] += 1
                        continue
                    # 外扩 25% 保棋子完整（含边缘光照），clip 到图边界
                    ex, ey = bw * 0.25, bh * 0.25
                    x0, y0 = max(0, int(bx - ex)), max(0, int(by - ey))
                    x1, y1 = min(w, int(bx + bw + ex)), min(h, int(by + bh + ey))
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        stat["裁片越界"] += 1
                        continue
                    patch = cv2.resize(img[y0:y1, x0:x1], (38, 38))
                    queues[cat2cls[a["category_id"]]].append(patch)
                    stat["候选"] += 1
                    stat[f"候选_{proj}"] += 1
    return queues, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_NPZ))
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="QC 底帧数上限（0=按需全量；冒烟用小值）")
    ap.add_argument("--min-conf", type=float, default=0.65)
    args = ap.parse_args()
    t0 = time.time()

    queues, stat = crop_boxes()
    total = sum(len(q) for q in queues.values())
    print(f"裁片候选：{total} 枚（{dict(stat)}）", flush=True)
    print(f"按类：{ {CLS_CHARS[i]: len(q) for i, q in sorted(queues.items())} }", flush=True)

    sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    lib: dict[int, list[np.ndarray]] = defaultdict(list)
    rng = random.Random(2027)
    n_qc = 0
    n_test = 0
    seed = 90_000
    # K/k 类永不出队（BAN 位保留渲染帅将）——while 条件必须排除，否则死循环
    # （v44 首跑实锤：剩余 2540 片全为 K/k，进程空转渲染永不结束）
    def _pending() -> bool:
        return any(len(q) for ci, q in queues.items() if CLS_CHARS[ci] not in BAN)

    while _pending():
        if args.limit_frames and n_qc >= args.limit_frames:
            break
        # 渲一帧底帧（要求 ≥24 占子格的整盘；opening/midgame 通常满足）
        s = None
        for _ in range(8):
            cand = R.make_sample(random.Random(seed), T.font_pools()[0])
            seed += 1
            if int((cand["labels"] > 0).sum()) >= 24:
                s = cand
                break
        n_qc += 1
        if s is None:
            continue
        bgr = cv2.cvtColor(s["cls_img"], cv2.COLOR_RGB2BGR)
        placed = []                                     # (cell, cls_idx, patch)
        cells = list(range(90))
        rng.shuffle(cells)
        for c in cells:
            v = int(s["labels"][c])
            if v <= 0 or v == IDX_DOT:
                continue
            ch = CLS_CHARS[v]
            if ch in BAN or not queues.get(v):
                continue                                # K/k 位保留渲染帅将
            patch = queues[v].popleft()
            cx, cy = AT.cls_cell_center(c // 9, c % 9)
            AT.paste_piece(bgr, patch, cx, cy)
            placed.append((c, IDX[ch], patch))
        if not placed:
            continue
        # nano 单帧推理（batch=1——BN 批污染坑）
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        x = np.ascontiguousarray(((x - T.IM_MEAN_NP) / T.IM_STD_NP)[None])
        lo = sess.run(None, {iname: x})[0][0]
        for c, ci, patch in placed:
            n_test += 1
            row = lo[c].astype(np.float64)
            row = row / max(1e-9, row.sum())
            if int(lo[c].argmax()) == ci and float(row[ci]) >= args.min_conf:
                lib[ci].append(patch)
        if n_qc % 100 == 0:
            print(f"  QC {n_qc} 帧，入库 {sum(len(v) for v in lib.values())}/{n_test}，"
                  f"队列余 {sum(len(q) for q in queues.values())}", flush=True)

    dist = {CLS_CHARS[i]: len(v) for i, v in sorted(lib.items())}
    n_lib = sum(len(v) for v in lib.values())
    print(f"QC 完成：{n_qc} 帧底帧，入库 {n_lib}/{n_test}（{n_lib/max(1,n_test):.0%}），"
          f"覆盖 {len(lib)} 类 {dist}，{time.time()-t0:.0f}s", flush=True)
    if not lib:
        raise SystemExit("VN 贴片库为空：检查映射/QC 阈值")
    # 落盘：按类数组（等长 pad 不做——npz 允许不等长 per-key 数组）
    out = {f"cls{i}": np.stack(v) for i, v in lib.items()}
    out["meta"] = np.array([json.dumps({
        "n_lib": n_lib, "n_test": n_test, "n_qc": n_qc,
        "min_conf": args.min_conf, "dist": dist,
    }, ensure_ascii=False)])
    np.savez_compressed(args.out, **out)
    print(f"落盘 {args.out}（{Path(args.out).stat().st_size/1e6:.1f}MB）")


if __name__ == "__main__":
    main()
