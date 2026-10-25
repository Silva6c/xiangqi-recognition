# -*- coding: utf-8 -*-
"""照片棋子贴图合成（v16~v17 计划 B3：实拍域外观桥）。

与 augment_real_transplant.py（v12）的分工：
- transplant：同帧屏幕棋子贴同帧空位——补特定格位组合（沉底炮等），同域零痕迹；
- 本脚本：**照片域棋子 patch 贴到屏幕帧**——把爬虫实拍图里的木雕/树脂棋子外观
  注入屏幕几何域，让模型在规整几何上见过实物棋子（反方向桥接实拍域）。

流程：
1. patch 库：real_photo_train.npz（爬虫伪标签训练集）中所有非空/非 IGNORE 格，
   按类裁圆形棋子 patch（含光照/纹理/字体多样性）；
2. 合成：屏幕帧（real_frames_labeled.npz）每帧随机选 2~5 个格位（棋子或空位），
   从 patch 库随机取异类 patch 羽化贴入，标签同步改；
3. teacher 质检：nano_v3 对贴片格软标签概率 ≥ 阈值且 argmax 命中目标类，否则丢弃。

产物 data/photo_composite.npz：imgs [M,315,280,3] u8 RGB + labels [M,90]。
用法：python augment_photo_composite.py [--per-frame 3] [--out data/photo_composite.npz]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import augment_real_transplant as AT
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
# v18：patch 库合并主训练集 + 两档回收集（741 张未入训爬虫图的难例外观）
PHOTO_NPZS = [HERE / "data/real_photo_train.npz",
              HERE / "data/real_photo_recover.npz",
              HERE / "data/real_photo_recover45.npz"]
SCREEN_NPZ = HERE / "data/real_frames_labeled.npz"
OUT_NPZ = HERE / "data/photo_composite.npz"

CLS = AT.CLS
IDX_DOT = CLS.index(".")


def build_patch_library(min_cell_conf_tag: float = 0.0) -> dict[int, list[np.ndarray]]:
    """从照片伪标签训练集裁棋子 patch，按类归档（v18 起合并主集 + 回收集）。

    labels 里 IGNORE=-100、'.'=空位；只收真实棋子格。回收集 board 门槛放宽到
    0.45/0.55，patch 质量参差——由 teacher 质检兜底（贴上认不出的自然被拒）。"""
    lib: dict[int, list[np.ndarray]] = {}
    n_total = 0
    for npz_path in PHOTO_NPZS:
        if not npz_path.exists():
            continue
        z = np.load(npz_path, allow_pickle=False)
        imgs, labs = z["imgs"], z["labels"]
        n_before = n_total
        for k in range(len(imgs)):
            bgr = cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR)
            lab = labs[k]
            for c in range(90):
                v = int(lab[c])
                if v < 0 or v == IDX_DOT:
                    continue
                patch = AT.circle_patch(bgr, *AT.cls_cell_center(c // 9, c % 9),
                                        radius=15)
                if patch is None:
                    continue
                lib.setdefault(v, []).append(patch)
                n_total += 1
        print(f"  {npz_path.name}: 累计 patch {n_total}（+{n_total - n_before}）")
    print(f"patch 库：{n_total} 个，覆盖 {len(lib)} 类 "
          f"({ {CLS[i]: len(v) for i, v in sorted(lib.items())} })")
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-frame", type=int, default=3, help="每帧平均贴片数")
    ap.add_argument("--variants", type=int, default=1,
                    help="每底帧独立贴片变体数（v18 用 3 扩量，底帧 450 的样本上限随之 ×N）")
    ap.add_argument("--min-teacher-conf", type=float, default=0.55)
    ap.add_argument("--out", default=str(OUT_NPZ))
    args = ap.parse_args()

    lib = build_patch_library()
    if not lib:
        raise SystemExit("patch 库为空：先跑 annotate_photo.py 生成 real_photo_train.npz")

    z = np.load(SCREEN_NPZ, allow_pickle=False)
    imgs, labs = z["imgs"], z["labels"]
    n_src = len(imgs)
    print(f"底帧：{n_src} 帧屏幕真实标注")

    sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rng = random.Random(20261003)

    out_imgs, out_labs = [], []
    n_try = n_pass = 0
    classes = sorted(lib)
    for k in range(n_src):
        bgr = cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR)
        lab = labs[k]
        for _v in range(args.variants):
            patched = bgr.copy()
            new_lab = lab.copy()
            n_paste = rng.randint(max(1, args.per_frame - 1), args.per_frame + 2)
            frame_pass = 0
            for _ in range(n_paste):
                # 目标格：任意可写格（棋子格=换子外观；空位=贴新子）
                ti = rng.randrange(90)
                src_cls = rng.choice(classes)             # 贴入的类别
                if len(lib[src_cls]) == 0:
                    continue
                patch = rng.choice(lib[src_cls])
                n_try += 1
                tcx, tcy = AT.cls_cell_center(ti // 9, ti % 9)
                trial = patched.copy()
                AT.paste_piece(trial, patch, tcx, tcy)
                x = cv2.cvtColor(trial, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
                x = ((x - T.IM_MEAN_NP) / T.IM_STD_NP)[None].astype(np.float32)
                lo = sess.run(None, {iname: np.ascontiguousarray(x)})[0][0]
                p_row = lo[ti].astype(np.float64)
                p_row = p_row / max(1e-9, p_row.sum())
                conf = float(p_row[src_cls])
                if conf < args.min_teacher_conf or int(lo[ti].argmax()) != src_cls:
                    continue
                patched = trial                            # 质检通过才落
                new_lab[ti] = src_cls
                n_pass += 1
                frame_pass += 1
            if frame_pass:
                out_imgs.append(cv2.cvtColor(patched, cv2.COLOR_BGR2RGB))
                out_labs.append(new_lab)
        if (k + 1) % 80 == 0:
            print(f"  {k + 1}/{n_src} 帧，通过 {n_pass}/{n_try}", flush=True)

    M = len(out_imgs)
    print(f"合成完成：尝试 {n_try}，teacher 质检通过 {n_pass}（{n_pass / max(1, n_try):.0%}）")
    np.savez_compressed(args.out, imgs=np.stack(out_imgs).astype(np.uint8),
                        labels=np.stack(out_labs).astype(np.int64))
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"落盘 {args.out}: {M} 样本, {size_mb:.0f}MB")


if __name__ == "__main__":
    main()
