# -*- coding: utf-8 -*-
"""评测集构建（自研识别引擎一期，docs/自研识别引擎方针.md 阶段1/2）。

三套评测集（落盘 data/eval_*.npz，可重复构建）：
1. eval_indist.npz    —— 渲染·同分布（训练字体池，固定种子 300 帧）：验证训练收敛质量；
2. eval_holdout.npz   —— 渲染·泛化（held-out 字体 + 更强扰动，300 帧）：没见过的字体风格；
3. eval_bench.npz     —— 真实帧（bench 实验 12 底片中可用的 11 张，RTMPose 定位 → warp）：
                           与 ONNX/YOLO 同口径的回归基准（cases.json 三源交叉真值）；
   附带负样本（abstain 评测用）：底片错位 warp（角点大偏移/角点置换）+ 底片盘外裁剪。

用法：python build_eval.py [--force]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import renderer as R
from pose_corner import PoseCornerDetector, warp_frame_to_warp450
from train import GEN_HOLDOUT, font_pools


def _to_npz(path: Path, imgs: list, labels: list, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, imgs=np.stack(imgs), labels=np.stack(labels), meta=json.dumps(meta, ensure_ascii=False))
    print(f"  {path.name}: {len(imgs)} 帧")


def build_rendered(name: str, seed: int, fonts: list, n: int, hard: bool) -> None:
    """渲染评测集。hard=True 时施加更强的泛化压力（更暗/更亮的极端、更强模糊）"""
    rng = random.Random(seed)
    imgs, labels, metas = [], [], []
    for i in range(n):
        s = R.make_sample(rng, fonts)
        if hard and rng.random() < 0.5:
            # 极端光度：复刻端上可能遇到的暗夜模式/强光过曝
            w = s["w450"]
            if rng.random() < 0.5:
                w = (w.astype(np.float32) * rng.uniform(0.45, 0.62)).astype(np.uint8)   # 暗光
            else:
                w = np.clip(w.astype(np.float32) * rng.uniform(1.5, 1.9) + 40, 0, 255).astype(np.uint8)  # 过曝
            s["cls_img"] = R.warp450_to_cls(w)
        imgs.append(s["cls_img"])
        labels.append(s["labels"])
        metas.append({"stage": s["stage"], "theme": s["theme"]})
    _to_npz(Path(__file__).parent / "data" / f"eval_{name}.npz", imgs, labels,
            {"seed": seed, "fonts": [Path(f).name for f in fonts], "hard": hard, "items": metas})


def build_bench(force: bool = False) -> None:
    """bench 底片 → 分类输入 + 真值标注 + 负样本（abstain 评测用）"""
    repo = Path(__file__).resolve().parent
    cases_path = repo / "testdata/recog-bench/cases.json"
    bench_dir = repo / ".shots/bench"
    if not cases_path.exists() or not bench_dir.exists():
        print("  [跳过] bench 底片或 cases.json 不存在（.shots 不入库，仅本机可建）")
        return
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    det = PoseCornerDetector()
    imgs, labels, ids, quads = [], [], [], []
    neg_shift, neg_offboard = [], []
    n_util = 0
    for case in cases:
        cid = case["id"]
        png = bench_dir / f"case{cid:02d}.png"
        if not png.exists():
            print(f"  [跳过] case{cid:02d} 底片缺失")
            continue
        frame = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [跳过] case{cid:02d} 底片读取失败")
            continue
        corners = det.detect_with_check(frame)
        if corners is None:
            print(f"  [跳过] case{cid:02d} 角点定位失败/不可信")
            continue
        # FEN（canonical，红在 bottom）→ 屏幕布局。cnvcs 对局不翻盘 = STANDARD：FEN 行序即屏幕行序
        fen_rows = case["fen"].split()[0].split("/")
        assert len(fen_rows) == 10
        lab = np.full(90, R.CLASS_INDEX["."], np.int64)
        for r, row in enumerate(fen_rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    lab[r * 9 + c] = R.CLASS_INDEX[ch]
                    c += 1
        w450 = warp_frame_to_warp450(frame, corners)          # TL,TR,BL,BR 顺序
        imgs.append(R.warp450_to_cls(w450))
        labels.append(lab)
        ids.append(cid)
        quads.append(corners.tolist())
        # 负样本 A：错位 warp（角点整体大偏移 40~90px，模拟 pose 复检前的错位采图）
        shift = rng2 = np.random.RandomState(9000 + cid)
        off = shift.uniform(40, 90, size=(4, 2)).astype(np.float32) * shift.choice([-1, 1], size=(4, 2))
        bad = w450_of(frame, corners + off)
        neg_shift.append(R.warp450_to_cls(bad))
        # 负样本 B：盘外裁剪（取底片左上角 280×315 区域，屏幕 UI 内容）
        neg_offboard.append(cv2.cvtColor(frame[0:315, 0:280], cv2.COLOR_BGR2RGB))
        n_util += 1
    if n_util == 0:
        print("  [跳过] 无可用底片")
        return
    _to_npz(Path(__file__).parent / "data" / "eval_bench.npz", imgs, labels,
            {"case_ids": ids, "quads": quads, "source": "bench 12 底片（case01 误拍主菜单不可用）"})
    _to_npz(Path(__file__).parent / "data" / "eval_bench_neg.npz", neg_shift + neg_offboard,
            [np.zeros(90, np.int64)] * len(neg_shift + neg_offboard),
            {"kinds": ["misaligned_warp"] * len(neg_shift) + ["offboard_crop"] * len(neg_offboard)})

def w450_of(frame, corners):
    return warp_frame_to_warp450(frame, corners)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    data_dir = Path(__file__).parent / "data"
    train_fonts, gen_fonts = font_pools()
    print(f"字体池：训练 {len(train_fonts)} / 泛化 {len(gen_fonts)}")
    if args.force or not (data_dir / "eval_indist.npz").exists():
        build_rendered("indist", 8001, train_fonts, 300, hard=False)
    if args.force or not (data_dir / "eval_holdout.npz").exists():
        build_rendered("holdout", 8002, gen_fonts, 300, hard=True)
    if args.force or not (data_dir / "eval_bench.npz").exists():
        build_bench()
    print("评测集构建完成")

if __name__ == "__main__":
    main()
