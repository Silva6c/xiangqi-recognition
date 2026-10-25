# -*- coding: utf-8 -*-
"""离线蒸馏集生成（v9 数据飞轮的下一级杠杆）。

原理：合成渲染样本经过全部姿态/光度增广（含 bench 型失真）后，现役 nano_v3 对
失真免疫（bench 100%），它输出的逐格概率分布就是"失真图像 → 正确局面"的软标签。
学生 GridHead 用 KL 蒸出 teacher 的失真不变性——这是有限真实帧给不了的覆盖密度。

架构（v10 起 GPU 化）：渲染进程池（CPU，多进程并行 make_sample）+
主进程 teacher 批量推理（CUDA EP，动态 batch 一次前向 32/64 帧）。
teacher 单帧 CPU 338~420ms 是原瓶颈；GPU 批量后吞吐预计 10 倍以上。

产物 data/distill_Nk.npz：
- imgs:   [N,315,280,3] uint8 RGB（合成+全增广，含 30% bench 型失真）
- probs:  [N,90,16]     float16 归一化 teacher 概率（软标签）
- labels: [N,90]        int64 合成真值（蒸馏槽里与 KL 混合的 hard 锚）

用法：
  python distill_gen.py --n 30000 --workers 12 --batch 64          # CUDA EP（自动检测）
  python distill_gen.py --provider cpu --n 12000 --workers 14      # 强制 CPU（原模式）
"""
from __future__ import annotations

import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import renderer as R
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"

# CUDA EP 依赖 CUDA 12 / cuDNN 9 运行库：本机未装系统级 CUDA，
# 借 torch（cu126）自带的 DLL——必须在 import onnxruntime 之前注入 PATH
if sys.platform == "win32":
    _torch_lib = Path(sys.executable).parent / "Lib/site-packages/torch/lib"
    if _torch_lib.exists():
        os.environ["PATH"] = str(_torch_lib) + os.pathsep + os.environ["PATH"]

SHARD_N = 6000  # 分片大小：控制主进程内存峰值（~1.6GB/片）

_W = {}  # 渲染 worker 进程内缓存


def _init_worker(fonts: list):
    _W["fonts"] = fonts


def _render_one(seed: int):
    """渲染 worker：确定性 seed 出（合成图, 合成真值），不做推理"""
    rng = random.Random(seed)
    s = R.make_sample(rng, _W["fonts"])
    return s["cls_img"], s["labels"].astype(np.int64)


def make_teacher():
    """主进程 teacher session：CUDA EP 优先（推理批量在主进程，显存单份）"""
    import onnxruntime as ort
    want_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    eps = ["CUDAExecutionProvider", "CPUExecutionProvider"] if want_cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(NANO_V3), providers=eps)
    return sess, sess.get_providers()[0]


def infer_batch(sess, iname: str, imgs: np.ndarray) -> np.ndarray:
    """批量软标签：[B,315,280,3] u8 → [B,90,16] float16 行归一化概率"""
    x = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    x = np.ascontiguousarray((x - T.IM_MEAN_NP) / T.IM_STD_NP)
    out = sess.run(None, {iname: x})[0]              # [B,90,16]（概率尺度，行和未必为 1）
    p = out.astype(np.float64)
    p = p / np.clip(p.sum(axis=2, keepdims=True), 1e-9, None)
    return p.astype(np.float16)


def save_shard(shards_dir: Path, idx: int, imgs: list, probs: list, labs: list) -> Path:
    out_p = shards_dir / f"shard_{idx:03d}.npz"
    np.savez(out_p, imgs=np.stack(imgs), probs=np.stack(probs), labels=np.stack(labs))
    return out_p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--workers", type=int, default=12, help="渲染进程数（CPU 渲染并行度）")
    ap.add_argument("--batch", type=int, default=64, help="teacher GPU 批量推理批大小")
    ap.add_argument("--provider", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed-base", type=int, default=9_000_000, help="蒸馏样本种子段（与训练渲染互不重叠）")
    ap.add_argument("--out", default=str(HERE / "data/distill_30k.npz"))
    args = ap.parse_args()

    train_fonts, _ = T.font_pools()
    sess, ep_name = make_teacher() if args.provider != "cpu" else (None, "CPUExecutionProvider")
    if args.provider == "cpu":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    print(f"字体池 {len(train_fonts)}；目标 {args.n} 样本 × {args.workers} 渲染进程，"
          f"teacher 批量推理 batch={args.batch} @ {ep_name}", flush=True)
    iname = sess.get_inputs()[0].name

    shards_dir = HERE / "data/distill_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []

    t0 = time.time()
    buf_imgs: list[np.ndarray] = []
    buf_probs: list[np.ndarray] = []
    buf_labs: list[np.ndarray] = []
    n_done = 0
    shard_idx = 0

    def flush_shard():
        nonlocal shard_idx, buf_imgs, buf_probs, buf_labs
        if not buf_imgs:
            return
        shard_paths.append(save_shard(shards_dir, shard_idx, buf_imgs, buf_probs, buf_labs))
        shard_idx += 1
        buf_imgs, buf_probs, buf_labs = [], [], []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(train_fonts,)) as ex:
        futs = [ex.submit(_render_one, args.seed_base + i) for i in range(args.n)]
        batch_imgs: list[np.ndarray] = []
        batch_labs: list[np.ndarray] = []
        for fu in as_completed(futs):
            img, lab = fu.result()
            batch_imgs.append(img)
            batch_labs.append(lab)
            if len(batch_imgs) < args.batch:
                continue
            probs = infer_batch(sess, iname, np.stack(batch_imgs))
            buf_imgs.extend(batch_imgs)
            buf_labs.extend(batch_labs)
            buf_probs.extend(probs)
            batch_imgs, batch_labs = [], []
            n_done += args.batch
            if len(buf_imgs) >= SHARD_N:
                flush_shard()
            if n_done % (args.batch * 20) == 0:
                rate = n_done / (time.time() - t0)
                eta = (args.n - n_done) / max(rate, 1e-6)
                print(f"  {n_done}/{args.n} ({rate:.1f} 样本/s, ETA {eta/60:.0f}min)", flush=True)
        # 尾批（不足 batch 的剩余样本）
        if batch_imgs:
            probs = infer_batch(sess, iname, np.stack(batch_imgs))
            buf_imgs.extend(batch_imgs)
            buf_labs.extend(batch_labs)
            buf_probs.extend(probs)
            n_done += len(batch_imgs)
        flush_shard()

    print(f"渲染+推理完成 {time.time() - t0:.0f}s（{n_done} 样本），合并 {len(shard_paths)} 分片…", flush=True)

    imgs_l, probs_l, labs_l = [], [], []
    for out_p in shard_paths:
        z = np.load(out_p)
        imgs_l.append(np.asarray(z["imgs"])); probs_l.append(np.asarray(z["probs"]))
        labs_l.append(np.asarray(z["labels"]))
        z.close()  # Windows：不关句柄 unlink 会失败
        out_p.unlink()
    np.savez_compressed(args.out, imgs=np.concatenate(imgs_l),
                        probs=np.concatenate(probs_l), labels=np.concatenate(labs_l))
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"落盘 {args.out}: {n_done} 样本, {size_mb:.0f}MB, 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
