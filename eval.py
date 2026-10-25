# -*- coding: utf-8 -*-
"""三方对拍评测（自研识别引擎一期，docs/自研识别引擎方针.md 阶段2）。

对比引擎：新模型 fp32 / 新模型 int8 / 旧 nano_v3-0319.onnx（现役主力）。
（YOLO 基线不在此跑：tflite 桌面口径失真，引用《识别引擎性能对比实验报告》的在机数据。）

评测集（build_eval.py 产出）：
- eval_indist.npz   渲染·同分布
- eval_holdout.npz  渲染·泛化（held-out 字体+极端光度）
- eval_bench.npz    bench 真实底片（11 帧，三源交叉真值）——方针的 100% 门槛口径
- eval_bench_neg.npz 负样本（错位 warp + 盘外裁剪）——弃权机制 ROC

指标：格子准确率 / 占子格准确率 / full-err-k / 单帧延迟（桌面 CPU 口径） /
弃权曲线（board 置信度阈值 vs 负样本拦截率 & 正样本保留率）。

用法：python eval.py [--model-dir runs/v1] [--out data/eval_report.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import renderer as R

CLS_INDEX = R.CLASS_INDEX
IGNORE = R.IGNORE_LABEL
NEG_META_KINDS = ["misaligned_warp", "offboard_crop"]


def load_set(path: Path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z["imgs"], z["labels"], meta


class Engine:
    def __init__(self, name: str, model_path: Path):
        self.name = name
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.path = model_path

    def infer(self, imgs_u8: np.ndarray, bs: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """返回 (argmax [N,90] int, 平均最大logit [N])——置信度口径与端上 avgLogit 一致。
        模型固定 batch=1（与端上一致），逐帧推理。"""
        n = len(imgs_u8)
        x = imgs_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        x = (x - IM_MEAN) / IM_STD
        preds = np.empty((n, 90), np.int64)
        confs = np.empty(n, np.float32)
        for i in range(n):
            lo = self.sess.run(None, {self.iname: np.ascontiguousarray(x[i:i+1])})[0]
            lo = lo.reshape(-1, 90, 16)
            preds[i] = lo.argmax(axis=2)[0]
            confs[i] = lo.max(axis=2).mean()
        return preds, confs

IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def accuracy(preds: np.ndarray, labels: np.ndarray) -> dict:
    valid = labels != IGNORE
    occ = valid & (labels != CLS_INDEX["."])
    ok = (preds == labels) & valid
    board_err = (~ok & valid).sum(axis=1)
    return {
        "boards": int(len(preds)),
        "cell_acc": float(ok.sum() / max(1, valid.sum())),
        "occ_acc": float((ok & occ).sum() / max(1, occ.sum())),
        "full_err0": float((board_err == 0).mean()),
        "full_err1": float((board_err <= 1).mean()),
        "full_err2": float((board_err <= 2).mean()),
        "board_err_hist": {int(k): int(v) for k, v in zip(*np.unique(board_err, return_counts=True))},
    }


def bench_engine_latency(eng: Engine, imgs: np.ndarray, rounds: int = 50) -> float:
    x = imgs[0:1].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    x = np.ascontiguousarray((x - IM_MEAN) / IM_STD)
    for _ in range(5):
        eng.sess.run(None, {eng.iname: x})
    t0 = time.time()
    for _ in range(rounds):
        eng.sess.run(None, {eng.iname: x})
    return (time.time() - t0) / rounds * 1000


def abstain_roc(pos_conf: np.ndarray, neg_conf: np.ndarray) -> list:
    """board 置信度阈值扫描：返回 [(thr, 正样本保留率, 负样本拦截率)]"""
    thrs = np.quantile(np.concatenate([pos_conf, neg_conf]), np.linspace(0.02, 0.98, 33))
    out = []
    for t in sorted(thrs):
        out.append((float(t), float((pos_conf >= t).mean()), float((neg_conf < t).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path(__file__).parent / "runs/v1"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "data/eval_report.json"))
    args = ap.parse_args()
    mdir = Path(args.model_dir)

    engines = [
        Engine("new_fp32", mdir / "gridhead_fp32.onnx"),
        Engine("new_int8", mdir / "gridhead_int8.onnx"),
        Engine("nano_v3", Path(__file__).parents[2] / "app/src/main/assets/models/nano_v3-0319.onnx"),
    ]

    data_dir = Path(__file__).parent / "data"
    report: dict = {"sets": {}, "latency_ms_desktop": {}, "abstain": {}}

    # v18 起补充三套：修复 start_layout 红背排 bug 后重建的渲染双集（无靶向偏置，
    # 与旧集并行保留做历史对照）+ 人工仲裁实拍真值集（IGNORE 格不罚）
    for set_name in ["indist", "holdout", "bench", "photo_crawl",
                     "indist_v18", "holdout_v18", "photo_human"]:
        p = data_dir / f"eval_{set_name}.npz"
        if not p.exists():
            print(f"[跳过] {p.name} 不存在（先跑 build_eval.py）")
            continue
        imgs, labels, meta = load_set(p)
        print(f"\n=== {set_name}（{len(imgs)} 帧）===")
        for eng in engines:
            preds, confs = eng.infer(imgs)
            m = accuracy(preds, labels)
            report["sets"].setdefault(set_name, {})[eng.name] = m
            print(f"  {eng.name:9s} cell={m['cell_acc']:.4f} occ={m['occ_acc']:.4f} "
                  f"err0={m['full_err0']:.3f} err1={m['full_err1']:.3f} err2={m['full_err2']:.3f}")
        report["latency_ms_desktop"][set_name] = {
            eng.name: round(bench_engine_latency(eng, imgs), 2) for eng in engines}

    # 弃权分析：bench 正样本 vs 负样本
    pb = data_dir / "eval_bench.npz"
    nb = data_dir / "eval_bench_neg.npz"
    if pb.exists() and nb.exists():
        imgs_p, _, _ = load_set(pb)
        imgs_n, _, _ = load_set(nb)
        for eng in engines:
            _, cp = eng.infer(imgs_p)
            _, cn = eng.infer(imgs_n)
            roc = abstain_roc(cp, cn)
            # 推荐阈值：负样本全拦的最低阈值（保留率即该阈值下正样本存活率）
            full_block = next((r for r in roc if r[2] >= 0.999), None)
            report["abstain"][eng.name] = {
                "roc": roc,
                "full_block_threshold": full_block,
                "note": "threshold=board 平均最大 logit；full_block=(阈值,正保留率,负拦截率)",
            }
            print(f"弃权[{eng.name}]: 负样本全拦阈值={full_block} 正样本保留率={full_block[1] if full_block else 'N/A'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n评测报告: {out}")

if __name__ == "__main__":
    main()
