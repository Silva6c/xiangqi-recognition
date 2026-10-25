# -*- coding: utf-8 -*-
"""第 1 波：RTMPose 伪标签生成（真实域角点训练弹药）+ 评测源排除清单 + heldout 切分。

数据源（全部为训练池源底片，非评测集）：
  - data/real_frames/*.png：266 张模拟器整屏（1080×2280 屏幕域）
  - data/photo_crawl*/raw|video_frames/*.jpg：~5300 张照片域原始帧
排除（红线）：
  - 全部 eval_* npz 的 files 字段逐一对照（26 帧金口径源图在内）
  - photo_crawl_eval/ 整目录隔离（评测补爬池）
流程：RTMPose detect_with_check（boxPlausible 同参过滤）→ 通过者入池 →
  photo 域随机 15%（上限 320）+ 屏幕域 30 张切 heldout（不训，用于一致率验收）。
输出：.cache/corner_pseudo.npz（files/corners TL,TR,BL,BR 原图坐标/split/domains）+
  .cache/corner_excl.json（排除清单审计留档）。
伪标签口径说明：RTMPose 在照片域（横图，旧窗全覆盖）与居中 App 屏可信；多棋盘照片
选盘歧义按"选中的也是真棋盘"接受（教定位不教选盘）。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pose_corner import PoseCornerDetector
from corner_data import imread_cn

EXCLUDE_NPZ = [
    "eval_photo_crawl.npz", "eval_photo_r2_pool.npz",
    "eval_photo_human.npz", "eval_photo_human_v2.npz",
    "eval_human_consensus49.npz", "eval_human_consensus68.npz",
    "eval_cells_hd.npz", "eval_bench.npz",
]
EXCLUDE_DIRS = {"photo_crawl_eval"}
SCREEN_DIRS = [HERE / "data/real_frames"]
PHOTO_DIRS = [d for d in (HERE / "data").glob("photo_crawl*")
              if d.is_dir() and d.name not in EXCLUDE_DIRS]

rng = random.Random(6410)
det = PoseCornerDetector()


def collect_excl() -> set:
    """eval_* npz 的 files 字段全收集（basename 匹配）。"""
    excl = set()
    for name in EXCLUDE_NPZ:
        p = HERE / "data" / name
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        if "files" in d.files:
            excl |= {Path(str(f)).name for f in d["files"]}
    # eval_bench 的 quads 涉及帧（meta JSON 内，files 缺失时兜底：源底片已丢，无帧可排）
    return excl


def iter_imgs():
    """yield (path, domain) 全部候选底片。"""
    for d in SCREEN_DIRS:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            yield p, "screen"
    for d in PHOTO_DIRS:
        for sub in ("raw", "video_frames", ""):
            sdir = d / sub if sub else d
            if not sdir.is_dir():
                continue
            for p in sorted(sdir.glob("*.jpg")) + sorted(sdir.glob("*.png")):
                yield p, "photo"


def main():
    excl = collect_excl()
    files, corners, splits, domains = [], [], [], []
    n_skipped_excl = n_failed_det = 0
    all_imgs = list(iter_imgs())
    print(f"候选底片 {len(all_imgs)} 张，排除清单 {len(excl)} 个文件名")
    for i, (p, dom) in enumerate(all_imgs):
        if p.name in excl:
            n_skipped_excl += 1
            continue
        img = imread_cn(p)
        if img is None or img.shape[0] < 200 or img.shape[1] < 200:
            n_failed_det += 1
            continue
        c = det.detect_with_check(img)
        if c is None:
            n_failed_det += 1
            continue
        files.append(str(p.relative_to(HERE)))
        corners.append(c)
        domains.append(dom)
        if i % 500 == 0:
            print(f"  [{i}/{len(all_imgs)}] 通过 {len(files)}", flush=True)
    corners = np.array(corners, np.float32)
    # heldout 切分：photo 域 15%（≤320）+ screen 域 30 张
    idx = list(range(len(files)))
    photo_idx = [i for i in idx if domains[i] == "photo"]
    screen_idx = [i for i in idx if domains[i] == "screen"]
    hold = set(rng.sample(photo_idx, min(len(photo_idx), 320))
               + rng.sample(screen_idx, min(len(screen_idx), 30)))
    splits = ["holdout" if i in hold else "train" for i in idx]
    np.savez_compressed(
        HERE / "../../.cache/corner_pseudo.npz",
        files=np.array(files), corners=corners,
        split=np.array(splits), domain=np.array(domains))
    json.dump({"exclude_names": sorted(excl), "exclude_npz": EXCLUDE_NPZ,
               "n_excl_skipped": n_skipped_excl, "n_det_failed": n_failed_det,
               "n_pass": len(files),
               "n_train": splits.count("train"), "n_holdout": splits.count("holdout")},
              open(HERE / "../../.cache/corner_excl.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"通过 {len(files)}/{len(all_imgs)}（排除跳过 {n_skipped_excl}，检测失败/过小 {n_failed_det}）"
          f"→ train {splits.count('train')} / holdout {splits.count('holdout')}")
    print("已落盘 .cache/corner_pseudo.npz + .cache/corner_excl.json")


if __name__ == "__main__":
    main()
