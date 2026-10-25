# -*- coding: utf-8 -*-
"""v46 efbok 分类贴片池直入管线（docs/v46方案-实物最后一格与bench攻坚-20261018.md §二A）。

背景：efbok 分类集 7190 张（10 类字面名，每类恰好 719，CC BY 4.0），整张图即一枚
棋子贴片——无需裁框，resize 38×38 直入 QC 队列（与 zfk/VN 同款开局标准位换子自证：
nano batch=1 双门 argmax==类 + conf≥0.65）。

类别映射与特殊处理：
 1-帅→K、9-将→k：BAN 源头不入队（防双帅，同主库规则）；
 2-仕→A、3-相→B、7-兵→P、8-卒→p、10-象→b：字面定色无歧义直入；
 4-马、5-炮、6-车：字面无红黑配对——**红黑双向自证**：patch 副本分别入红队
   （N/C/R）与黑队（n/c/r），QC 谁过归谁；红黑双过（字形歧义）与双不过（坏片）
   收尾统一弃，单过归正确类。

预处理损伤注意：Roboflow 导出已 Stretch 到 640×640（棋子微椭圆），nano QC 会自然
拦掉认不出的，通过率预期低于 VN 的 83% 基线。

产出 data/efbok_patch_lib.npz，供 augment_roboflow_patch.py --extra-lib 合并
（v46 起支持逗号分隔多库）。

用法：
  python augment_efbok_patch.py --limit-frames 40   # 冒烟
  python augment_efbok_patch.py                     # 全量 7190 张
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import augment_real_transplant as AT
import renderer as R
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
ROOT = (HERE.parent.parent.parent
        / "data/roboflow_raw/classification/chinese-chess-classification-efbok/v1")
OUT_NPZ = HERE / "data/efbok_patch_lib.npz"
BAN = ("K", "k")

CLS_CHARS = ".xKABNRCPkabnrcp"
IDX = {ch: i for i, ch in enumerate(CLS_CHARS)}

# 字面目录名 → 处置：单类字符直入；("amb", 红, 黑) 走红黑双向自证
DIR2CLS = {
    "1-帅": "K", "9-将": "k",                      # BAN
    "2-仕": "A", "3-相": "B", "7-兵": "P", "8-卒": "p", "10-象": "b",
    "4-马": ("amb", "N", "n"), "5-炮": ("amb", "C", "c"), "6-车": ("amb", "R", "r"),
}


def load_patches() -> tuple[dict[int, deque], Counter, dict[str, str]]:
    """遍历分类目录 → 整图 resize 38×38 按类入队。返回 (queues, stat, amb_md5)。"""
    queues: dict[int, deque] = defaultdict(deque)
    stat = Counter()
    amb_md5: dict[str, str] = {}                    # patch md5 → 源目录名（仅歧义源）
    seen: set[str] = set()
    for split in ("train", "valid", "test"):
        for d in sorted(ROOT.glob(f"{split}/*")):
            disp = DIR2CLS.get(d.name)
            if disp is None:
                stat["类外目录"] += 1
                continue
            for ip in sorted(d.glob("*.jpg")):
                md5 = hashlib.md5(ip.read_bytes()).hexdigest()
                if md5 in seen:
                    stat["md5重复图"] += 1
                    continue
                seen.add(md5)
                # cv2.imread 在 Windows 上打不开含中文目录名的路径（ANSI 编码限制），
                # 改 np.fromfile+imdecode 内存解码
                img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    stat["读取失败"] += 1
                    continue
                patch = cv2.resize(img, (38, 38))   # 整图即贴片（640×640 Stretch 已知）
                if isinstance(disp, str):
                    if disp in BAN:
                        stat["KkBAN"] += 1
                        continue
                    queues[IDX[disp]].append(patch)
                    stat["候选_单义"] += 1
                else:
                    _, red, blk = disp
                    pmd5 = hashlib.md5(patch.tobytes()).hexdigest()
                    if pmd5 in amb_md5:
                        stat["md5重复片"] += 1
                        continue
                    amb_md5[pmd5] = d.name
                    queues[IDX[red]].append(patch)  # 双向自证：副本分别入红黑队
                    queues[IDX[blk]].append(patch)
                    stat["候选_歧义"] += 1
    return queues, stat, amb_md5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_NPZ))
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="QC 底帧数上限（0=按需全量；冒烟用小值）")
    ap.add_argument("--min-conf", type=float, default=0.65)
    args = ap.parse_args()
    t0 = time.time()

    queues, stat, amb_md5 = load_patches()
    total = sum(len(q) for q in queues.values())
    print(f"裁片候选：{total} 枚（{dict(stat)}）", flush=True)
    print(f"按类：{ {CLS_CHARS[i]: len(q) for i, q in sorted(queues.items())} }", flush=True)

    sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    lib: dict[int, list[np.ndarray]] = defaultdict(list)
    amb_pass: dict[str, list[int]] = defaultdict(list)   # md5 → 通过的类（红黑双过去重用）
    rng = random.Random(2027)
    n_qc = 0
    n_test = 0
    seed = 97_000

    while sum(len(q) for q in queues.values()):
        if args.limit_frames and n_qc >= args.limit_frames:
            break
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
        placed = []
        cells = list(range(90))
        rng.shuffle(cells)
        for c in cells:
            v = int(s["labels"][c])
            if v <= 0 or v == IDX["."] or not queues.get(v):
                continue
            patch = queues[v].popleft()
            cx, cy = AT.cls_cell_center(c // 9, c % 9)
            AT.paste_piece(bgr, patch, cx, cy)
            placed.append((c, v, patch))
        if not placed:
            continue
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        x = np.ascontiguousarray(((x - T.IM_MEAN_NP) / T.IM_STD_NP)[None])
        lo = sess.run(None, {iname: x})[0][0]
        for c, ci, patch in placed:
            n_test += 1
            row = lo[c].astype(np.float64)
            row = row / max(1e-9, row.sum())
            if int(lo[c].argmax()) == ci and float(row[ci]) >= args.min_conf:
                lib[ci].append(patch)
                pmd5 = hashlib.md5(patch.tobytes()).hexdigest()
                if pmd5 in amb_md5:
                    amb_pass[pmd5].append(ci)
        if n_qc % 100 == 0:
            print(f"  QC {n_qc} 帧，入库 {sum(len(v) for v in lib.values())}/{n_test}，"
                  f"队列余 {sum(len(q) for q in queues.values())}", flush=True)

    # 歧义源收尾：红黑双过（同 md5 两类都过=字形歧义）与从未单过的不确定片，统一弃
    n_drop_dual = 0
    for pmd5, cis in amb_pass.items():
        if len(set(cis)) >= 2:
            n_drop_dual += 1
            for ci in set(cis):
                keep = [p for p in lib[ci]
                        if hashlib.md5(p.tobytes()).hexdigest() != pmd5]
                lib[ci] = keep
    dist = {CLS_CHARS[i]: len(v) for i, v in sorted(lib.items())}
    n_lib = sum(len(v) for v in lib.values())
    print(f"歧义收尾：红黑双过弃 {n_drop_dual} 枚（单过归位 {len(amb_pass)-n_drop_dual}）")
    print(f"QC 完成：{n_qc} 帧底帧，入库 {n_lib}/{n_test}（{n_lib/max(1,n_test):.0%}），"
          f"覆盖 {len(lib)} 类 {dist}，{time.time()-t0:.0f}s", flush=True)
    if not lib:
        raise SystemExit("efbok 贴片库为空：检查映射/QC 阈值")
    out = {f"cls{i}": np.stack(v) for i, v in lib.items()}
    out["meta"] = np.array([json.dumps({
        "n_lib": n_lib, "n_test": n_test, "n_qc": n_qc,
        "min_conf": args.min_conf, "dist": dist, "stat": dict(stat),
        "n_amb_src": len(amb_md5), "n_dual_pass_drop": n_drop_dual,
    }, ensure_ascii=False)])
    np.savez_compressed(args.out, **out)
    print(f"落盘 {args.out}（{Path(args.out).stat().st_size/1e6:.1f}MB）")


if __name__ == "__main__":
    main()
