# -*- coding: utf-8 -*-
"""实拍照片伪标签管线（v16~v17 计划 B1/B2）：爬虫图 + 视频帧 → 训练/评测 npz。

链路与 annotate_real.py 相同：RTMPose 角点（detect_with_check 可信校验）→ warp450
→ 中心裁剪 280x315 → nano_v3 两域投票伪标签（≥2 域一致 + 置信度门槛）。

与屏幕域版本的三处关键差异：
1. 不做局面序列去重——实拍图的价值在外观多样性（木雕字/光照/棋子样式），
   开局盘面预测序列相同是常态，按局面去重会把外观多样性全删掉（源图级
   aHash 去重已由爬虫完成）。
2. 按源图确定性切分 holdout：每第 1/round(1/frac) 张进评测集，其余进训练集；
   同一源图的两域样本永不跨集（防泄漏）。
3. 全部通过帧落诊断拼图（两域并排）到 data/photo_crawl/diag/，供人工抽查校准
   评测集标签（B4）。

输出：
- data/real_photo_train.npz  imgs[N,315,280,3]u8 + labels[N,90] + bconf + files + meta
- data/eval_photo_crawl.npz  同结构（评测 holdout，标签待 B4 人工校准后使用）
用法：
  python annotate_photo.py --diag        # 先看 pose 成功率与置信度分布
  python annotate_photo.py --holdout-frac 0.18
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import pose_corner as PC

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
PHOTO_DIR = HERE / "data/photo_crawl"
OUT_TRAIN = HERE / "data/real_photo_train.npz"
OUT_EVAL = HERE / "data/eval_photo_crawl.npz"

IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

# 两域投票（与 annotate_real 同参：原生域 + bench 型失真域）
SCALE_CORRECTIONS = (((1.0, 1.0), (0.0, 0.0)), ((1.029, 1.005), (23.8, 9.5)))

# 16 类顺序 . x KABNRCP kabnr cp 与 GLYPHS 对照，诊断拼图标注用
CLS2CH = ".×KABNRCPkabnrcp"


def shrink_corners(corners: np.ndarray, spec) -> np.ndarray:
    (sx, sy), (dx, dy) = spec
    c = corners.mean(axis=0, keepdims=True)
    out = c + (corners - c)
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out + np.array([dx, dy], np.float32)


def warp450_to_cls(warp450_bgr: np.ndarray) -> np.ndarray:
    crop = warp450_bgr[25:475, 25:425]
    img = cv2.resize(crop, (280, 315), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def collect_sources() -> list[Path]:
    """爬虫 raw 图 + 视频帧，合并成一个稳定排序的源图清单。

    视频帧每 5s 一张、同视频内高度相似，先做 aHash 序列去重
    （与前一保留帧汉氏距离 <10 视为重复），避免单一视频过采样。"""
    import hashlib
    from PIL import Image

    files: list[Path] = []
    files += sorted((PHOTO_DIR / "raw").glob("*.jpg"))
    kept: list[Path] = []
    prev_hash = None
    for f in sorted((PHOTO_DIR / "video_frames").glob("*.jpg")):
        try:
            with Image.open(f) as im:
                g = im.convert("L").resize((8, 8), Image.LANCZOS)
                px = list(g.getdata())
            avg = sum(px) / len(px)
            hs = tuple(p > avg for p in px)
        except Exception:
            continue
        if prev_hash is not None and sum(a != b for a, b in zip(hs, prev_hash)) < 10:
            continue
        prev_hash = hs
        kept.append(f)
    files += kept
    return files


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 上不支持中文路径，改用 fromfile+imdecode。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: Path, img_bgr: np.ndarray, quality: int = 85) -> None:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-thr", type=float, default=0.5)
    ap.add_argument("--board-thr", type=float, default=0.7,
                    help="实拍域 teacher 置信整体低于屏幕域，默认收紧到 0.7")
    ap.add_argument("--holdout-frac", type=float, default=0.18,
                    help="评测集占比（按源图索引确定性切分）")
    ap.add_argument("--max-side", type=int, default=2000,
                    help="过大的源图先缩边（RTMPose 全图仿射对超大图不利）")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--only-missing", action="store_true",
                    help="增量回收：跳过已出现在现有 train/eval npz 里的源图")
    ap.add_argument("--out-train", default=None, help="训练集输出路径覆盖（回收模式必填）")
    ap.add_argument("--out-eval", default=None, help="评测集输出路径覆盖")
    ap.add_argument("--all-train", action="store_true",
                    help="全部进训练集不切评测（回收集专用，评测集保持冻结）")
    ap.add_argument("--photo-dir", default=None,
                    help="源图目录覆盖（默认 data/photo_crawl；评测补爬池用独立目录）")
    args = ap.parse_args()

    if args.photo_dir:
        global PHOTO_DIR
        PHOTO_DIR = Path(args.photo_dir)

    used: set[str] = set()
    if args.only_missing:
        # 防泄漏对照：主训练/评测集 + 两档回收集（回收集源图不进评测池）
        for p in (OUT_TRAIN, OUT_EVAL, HERE / "data/real_photo_recover.npz",
                  HERE / "data/real_photo_recover45.npz"):
            if p.exists():
                with np.load(p, allow_pickle=False) as z:
                    used.update(str(x) for x in z["files"])
        if args.out_train is None:
            raise SystemExit("拒绝执行：--only-missing 必须显式指定 --out-train，防止覆盖主训练集")

    frames = collect_sources()
    if used:
        n_all = len(frames)
        frames = [f for f in frames if f.name not in used]
        print(f"增量回收：{n_all} → {len(frames)} 源图（跳过已入训 {n_all - len(frames)}）")
    n_raw = sum(1 for f in frames if "raw" in f.parts)
    n_vid = len(frames) - n_raw
    print(f"待标注 {len(frames)} 源图（raw {n_raw} + video {n_vid}）")

    pose = PC.PoseCornerDetector()
    cls = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    cls_in = cls.get_inputs()[0].name

    tr = {"imgs": [], "labels": [], "bconf": [], "files": []}
    ev = {"imgs": [], "labels": [], "bconf": [], "files": []}
    diag_dir = PHOTO_DIR / "diag"
    if not args.diag:
        diag_dir.mkdir(exist_ok=True)

    pose_fail: list[str] = []
    low_board: list[str] = []
    all_cell_conf: list[float] = []
    all_board_conf: list[float] = []
    n_holdout = max(1, round(1 / args.holdout_frac))

    for idx, f in enumerate(frames):
        frame = imread_unicode(f)
        if frame is None:
            pose_fail.append(f.name + ":read_fail")
            continue
        h, w = frame.shape[:2]
        if max(h, w) > args.max_side:
            s = args.max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        corners = pose.detect_with_check(frame)
        if corners is None:
            pose_fail.append(f.name + ":pose_fail")
            continue

        domain_imgs, domain_preds, domain_conf = [], [], []
        for sc in SCALE_CORRECTIONS:
            warp = PC.warp_frame_to_warp450(frame, shrink_corners(corners, sc))
            cls_img = warp450_to_cls(warp)
            x = cls_img.astype(np.float32).transpose(2, 0, 1) / 255.0
            x = np.ascontiguousarray((x - IM_MEAN) / IM_STD)[None]
            logits = cls.run(None, {cls_in: x})[0].reshape(90, 16)
            domain_imgs.append(cls_img)
            domain_preds.append(logits.argmax(axis=1))
            domain_conf.append(logits.max(axis=1))

        stack = np.stack(domain_preds)
        mode = np.zeros(90, np.int64)
        votes = np.zeros(90, np.int64)
        for j in range(90):
            cnt = np.bincount(stack[:, j], minlength=16)
            mode[j] = cnt.argmax()
            votes[j] = cnt.max()
        agree = votes >= 2
        pred = mode
        cell_conf = np.min(np.stack(domain_conf), axis=0).copy()
        cell_conf[~agree] = 0.0
        bconf = float(cell_conf.mean())
        all_cell_conf.extend(cell_conf.tolist())
        all_board_conf.append(bconf)

        if bconf < args.board_thr:
            low_board.append(f"{f.name}:{bconf:.2f}")
            continue

        labels = pred.astype(np.int64).copy()
        labels[~agree] = -100
        labels[cell_conf < args.cell_thr] = -100

        if not args.diag:
            # 诊断拼图：两域并排（评测集与训练集都落，人工抽查共用）
            pair = np.concatenate(domain_imgs, axis=1)[:, :, ::-1]  # RGB→BGR
            imwrite_unicode(diag_dir / f"{f.stem}.jpg", pair)
            bucket = tr if (args.all_train or idx % n_holdout != 0) else ev
            for dimg in domain_imgs:
                bucket["imgs"].append(dimg)
                bucket["labels"].append(labels)
                bucket["bconf"].append(bconf)
                bucket["files"].append(f.name)

    n_src = len(frames)
    print(f"\npose 失败 {len(pose_fail)}/{n_src} "
          f"({len(pose_fail) / max(1, n_src) * 100:.1f}%): {pose_fail[:6]}")
    print(f"board<{args.board_thr} 丢弃 {len(low_board)} 帧: {low_board[:6]}")
    q = np.quantile(all_cell_conf, [0.05, 0.5, 0.95])
    print(f"逐格 logit 分位: p5={q[0]:.2f} p50={q[1]:.2f} p95={q[2]:.2f}")
    qb = np.quantile(all_board_conf, [0.05, 0.5, 0.95])
    print(f"整帧 logit 分位: p5={qb[0]:.2f} p50={qb[1]:.2f} p95={qb[2]:.2f}")

    if args.diag:
        return

    out_train = Path(args.out_train) if args.out_train else OUT_TRAIN
    out_eval = Path(args.out_eval) if args.out_eval else OUT_EVAL
    for path, bucket in ((out_train, tr), (out_eval, ev)):
        if not bucket["imgs"]:
            continue
        np.savez_compressed(
            path,
            imgs=np.stack(bucket["imgs"]),
            labels=np.stack(bucket["labels"]),
            bconf=np.array(bucket["bconf"], np.float32),
            files=np.array(bucket["files"]),
            meta=json.dumps({
                "cell_thr": args.cell_thr, "board_thr": args.board_thr,
                "holdout_frac": args.holdout_frac,
                "source": "爬虫图（百度/必应图片搜索）+ B 站视频抽帧，"
                          "aHash 去重后 pose 可信过滤，nano_v3 两域投票伪标签",
                "annotator": "nano_v3-0319 两域投票",
                "eval_note": "eval_photo_crawl 标签为伪标签，须经 B4 人工抽查校准"
            }, ensure_ascii=False),
        )
        lab = np.stack(bucket["labels"])
        print(f"落盘 {path.name}: {len(bucket['imgs'])} 样本"
              f"（源图 {len(set(bucket['files']))}，IGNORE 格 "
              f"{np.mean(lab == -100) * 100:.1f}%）")


if __name__ == "__main__":
    main()
