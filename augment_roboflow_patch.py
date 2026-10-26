# -*- coding: utf-8 -*-
"""v43 实物贴片管线（docs/v43实物贴片方案-20261017.md §2.2，v42 整帧真监督证伪后的路线修正）。

核心思想：贴片只依赖「框内图像+类别」（COCO 人工标注保证，对标签漏标天然免疫），
位置语义由渲染端标准格位成立（几何一致性天然满足）——正打实物域 8 倍差距
（26 帧实物域 v41 24 错 vs nano 3 错，蒸馏天花板）。

QC 语义（冒烟修订，对方案 §2.2 的偏离记录）：
 原设计在「渲染帧+贴片」上 QC 双门（conf≥0.55 + argmax==目标类）。冒烟实测通过率 0/67——
 根因是 nano 的格位先验在裁决（红仕贴黑区被按先验认成黑仕/认空），而打破格位先验恰恰是
 贴片路线的目的（路边摊红子随处摆），QC 等于让先验否决路线本身。修正为**原帧自证**：
 贴片在它来源的格位上（无先验冲突——子本来就摆在那）由 nano（强域老师）验证 argmax==COCO 类
 且 conf 达标，坏贴片（叠压/裁废/标错类）在库端全拦；渲染帧上不再 QC——贴片+标签几何一致，
 nano 在渲染帧上认不出是 nano 的事，正是要教的监督信号。

流程：
 1. nano_judge 一次批推理（teacher=nano 单师，帧级+格级同源）：
    帧级机剔——一致率 <0.5 ∪ 占子格漏检率 >0.8 的帧不进源（剔「堆叠摆拍」帧，
    贴片互相叠压裁出来是坏的；v42 实测 32.7% 格不一致、零干净帧，机剔为第二道防线）；
    格级自证——占子格 nano argmax==COCO 类且 conf≥0.55 才可裁片入库；
 2. 贴片库：幸存帧合格格按「渲染格心 + 框心偏移(offsets)」裁圆形贴片（radius 15 → 38×38
    窗口，贴上直径 30 与渲染棋子直径 ~28 匹配；按框心裁根治 σ≈10-13px 偏心裁剪），
    按 16 类归档，去 K/k（渲染帧已有帅将，再贴成双帅）；
 3. 底帧：renderer.make_sample 确定性 seed 离线渲染（训练字体池 + 全增广，与训练同分布）；
 4. 贴：每帧随机 2~4 个空位（label=='.'，互不相邻防圆片叠压），库内随机类+随机片羽化贴入，
    标签同步；库端已质检，此步无 QC；
 5. 分片落盘 real_rf_patch_transplant_shard{i}.npz（train --real-npz 逗号并列即用）。

用法：
  python augment_roboflow_patch.py                      # 全量 1.2 万底帧
  python augment_roboflow_patch.py --frames 24 --out data/pools --tag smoke_rfpt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import augment_real_transplant as AT
import renderer as R
import train as T

HERE = Path(__file__).resolve().parent
NANO_V3 = HERE.parents[1] / "app/src/main/assets/models/nano_v3-0319.onnx"
SRC_NPZ = HERE / "data/real_roboflow_sup.npz"
BASE_SEED = 20261017
CLS = AT.CLS                       # ".xKABNRCPkabnrcp"
IDX_DOT = CLS.index(".")
BAN_CLASSES = ("K", "k")           # 防双帅：库与贴片计划都排除

_TRAIN_FONTS: list | None = None   # worker 进程内的字体池（initializer 填充）


def _init_worker():
    """渲染 worker 初始化：训练字体池（与 train.py 完全同源）"""
    global _TRAIN_FONTS
    _TRAIN_FONTS = T.font_pools()[0]


def _render_chunk(bounds: tuple[int, int]) -> list:
    """渲染 [lo,hi) 区间底帧；每帧独立 rng（BASE_SEED+i）保证可复现"""
    lo, hi = bounds
    out = []
    for i in range(lo, hi):
        s = R.make_sample(random.Random(BASE_SEED + i), _TRAIN_FONTS)
        out.append((i, s["cls_img"], s["labels"]))
    return out


def nano_judge(imgs: np.ndarray, labs: np.ndarray, sess: ort.InferenceSession,
               min_agree: float, max_miss: float, min_conf: float):
    """teacher=nano 一次批推理，产出格级自证门（帧级仅统计报告，不剔帧）。

    帧级门退役的实测依据（冒烟）：nano 视角 99.7% 帧 agree<0.5（COCO 漏标把 agree 打穿，
    与"叠压摆拍"无关；方案 §2.3 引用的 26% 是 v41 口径——v41 与 COCO 标签同源偏差虚高）。
    贴片对漏标免疫，格级门（该格 nano 认对 + conf 达标）已完整覆盖帧级门要拦的叠压格。

    返回 (keep_frames [N] bool 仅统计用, pass_cells [N,90] bool 库准入)：
    - 帧级统计：一致率 <min_agree ∪ 占子格漏检率 >max_miss 的帧计数；
    - 格级：占子格 nano argmax==COCO 类且 conf≥min_conf 才可裁片入库（原帧自证）。
      conf 归一 = row/row.sum()（与 photo_composite 同款；nano 输出接近已归一概率，
      按 logits 做 softmax 会把概率再压一遍、conf 0.8 崩成 0.1——冒烟实测的坑）。

    推理必须逐帧 batch=1（冒烟实测的第二个坑）：nano 导出含未折叠的批统计算子，
    batch>1 时批内样本互相污染（同帧 logits 差 0.87、全部占子格输出'.'）——
    端上真实工作模式就是单帧，teacher 判定必须模拟端上口径。photo_composite
    时代全程单帧循环从未踩坑；v43 为提速改批推理两次全灭的根因即此。"""
    iname = sess.get_inputs()[0].name
    keep = np.zeros(len(imgs), bool)
    pass_cells = np.zeros(labs.shape, bool)
    n_cell_all = n_cell_pass = 0
    x_std = (T.IM_MEAN_NP, T.IM_STD_NP)
    for k in range(len(imgs)):
        x = imgs[k].astype(np.float32).transpose(2, 0, 1) / 255.0
        x = np.ascontiguousarray(((x - x_std[0]) / x_std[1])[None])
        lo = sess.run(None, {iname: x})[0][0]                    # [90,16]
        pred = lo.argmax(axis=1)                                 # [90]
        prob = lo / np.maximum(1e-9, lo.sum(axis=1, keepdims=True))
        lab = labs[k]
        agree = float((pred == lab).mean())
        occ = lab != IDX_DOT
        occ_n = int(occ.sum())
        miss = float((occ & (pred == IDX_DOT)).sum()) / max(1, occ_n)
        keep[k] = (agree >= min_agree) & (miss <= max_miss)
        lab_safe = np.clip(lab, 0, None)
        conf_self = prob[np.arange(90), lab_safe]
        pass_cells[k] = occ & (pred == lab) & (conf_self >= min_conf)
        n_cell_all += occ_n
        n_cell_pass += int(pass_cells[k].sum())
        if (k + 1) % 500 == 0:
            print(f"  teacher 判定 {k + 1}/{len(imgs)}，格级累计通过 {n_cell_pass}",
                  flush=True)
    print(f"teacher 判定：帧级统计（不剔帧）——agree≥{min_agree} 且 miss≤{max_miss} 的帧 "
          f"{int(keep.sum())}/{len(imgs)}（COCO 漏标污染 agree，仅报告）；"
          f"格级自证 {n_cell_pass}/{n_cell_all} 占子格通过"
          f"（{n_cell_pass / max(1, n_cell_all):.0%}，库唯一准入）", flush=True)
    return keep, pass_cells


def torch_no_grad():
    """（已废弃占位，保留防外部引用）"""
    from contextlib import nullcontext
    return nullcontext()


def build_patch_library(imgs: np.ndarray, labs: np.ndarray, offs_all: np.ndarray,
                        pass_cells: np.ndarray,
                        radius: int) -> dict[int, list[np.ndarray]]:
    """贴片库：格级自证合格格（全帧范围，不按帧剔——帧级门已被实测退役），
    按「渲染格心 + 框心偏移」裁圆片，按类归档（去 K/k）。
    框心 cls 坐标 = cls_cell_center(i,j) + offsets[k]；offsets 为 0 的格退回格心裁。"""
    lib: dict[int, list[np.ndarray]] = {}
    n_total = 0
    for k in range(len(imgs)):
        bgr = cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR)
        lab = labs[k]
        offs = offs_all[k]
        for c in range(90):
            v = int(lab[c])
            if v < 0 or v == IDX_DOT or CLS[v] in BAN_CLASSES or not pass_cells[k, c]:
                continue
            cx, cy = AT.cls_cell_center(c // 9, c % 9)
            cx += int(round(float(offs[c, 0])))
            cy += int(round(float(offs[c, 1])))
            patch = AT.circle_patch(bgr, cx, cy, radius=radius)
            if patch is None:
                continue
            lib.setdefault(v, []).append(patch)
            n_total += 1
    print(f"贴片库：{n_total} 枚，覆盖 {len(lib)} 类 "
          f"{ {CLS[i]: len(v) for i, v in sorted(lib.items())} }", flush=True)
    return lib


def pick_paste_plan(lab: np.ndarray, rng: random.Random,
                    classes: list[int], n_paste: int,
                    n_replace: int = 0) -> list[tuple[int, int]]:
    """选贴片计划：空位（'.'）且互不相邻（切比雪夫距离 ≥2 防圆片叠压邻格）。
    返回 [(cell_idx, class_idx)]；类均匀随机、具体片在贴时取。
    注意：贴片类与格位先验冲突是特性不是 bug（路边摊红子随处摆，正是要教的监督信号）。
    v58 密排变体：n_replace>0 时再选若干已占格（帅/将除外，同样距离 ≥2 约束），
    用真实裁片**替换渲染棋子**并改标签为贴片类（视觉=标签=真子，无错标；真子密度
    逼近实物盘的摆子密度，直打四轴 piece/roboflow 轴的真实形态覆盖）。"""
    def far(c: int, chosen: list[int]) -> bool:
        i, j = c // 9, c % 9
        return all(max(abs(i - x // 9), abs(j - x % 9)) >= 2 for x in chosen)

    cands = [c for c in range(90) if int(lab[c]) == IDX_DOT]
    rng.shuffle(cands)
    chosen: list[int] = []
    for c in cands:
        if far(c, chosen):
            chosen.append(c)
            if len(chosen) >= n_paste:
                break
    plan = [(c, rng.choice(classes)) for c in chosen]
    if n_replace > 0:
        occ = [c for c in range(90)
               if 0 <= int(lab[c]) != IDX_DOT and CLS[int(lab[c])] not in BAN_CLASSES]
        rng.shuffle(occ)
        n = 0
        for c in occ:
            if far(c, chosen):
                chosen.append(c)
                plan.append((c, rng.choice(classes)))
                n += 1
                if n >= n_replace:
                    break
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_NPZ),
                    help="roboflow_sup npz（需含 offsets 键——本版 parse_roboflow_coco.py 产出）")
    ap.add_argument("--out", default=str(HERE / "data"), help="分片输出目录")
    ap.add_argument("--frames", type=int, default=12000, help="渲染底帧数")
    ap.add_argument("--per-frame", type=int, default=3, help="每帧平均贴片数（实际 2~4 随机）")
    ap.add_argument("--radius", type=int, default=15,
                    help="贴片裁剪半径（patch 边=2*(r+4)；15→38px 窗口/贴上直径 30px）")
    ap.add_argument("--min-teacher-conf", type=float, default=0.65,
                    help="格级自证：nano 在原帧对贴片格目标类的 conf 下限"
                         "（0.65 实测拦掉叠压帧 tiles 坏片，库 -7.4%，目检复核）")
    ap.add_argument("--min-agree", type=float, default=0.5, help="帧级机剔：一致率下限")
    ap.add_argument("--max-miss", type=float, default=0.8, help="帧级机剔：占子格漏检率上限")
    ap.add_argument("--shard-n", type=int, default=6000, help="每分片帧数")
    ap.add_argument("--workers", type=int, default=8, help="渲染进程数")
    ap.add_argument("--tag", default="real_rf_patch_transplant", help="分片文件名前缀")
    ap.add_argument("--mode", choices=["sparse", "dense"], default="sparse",
                    help="v58：sparse=空位贴（原行为）；dense=空位贴+已占格替换（真子密排）")
    ap.add_argument("--replace-n", type=int, default=3,
                    help="dense 模式每帧替换已占格数（实际 ±1 随机；贴真子并改标签）")
    ap.add_argument("--extra-lib", default=str(HERE / "data/vn_raw_patch_lib.npz"),
                    help="额外贴片库 npz，逗号分隔可多个（VN/zfk/efbok 原图裁片；空串=不用）")
    args = ap.parse_args()

    t0 = time.time()

    # ---- 1) 载源 + teacher 判定（帧级机剔 + 格级自证）----
    z = np.load(args.src, allow_pickle=False)
    if "offsets" not in z:
        raise SystemExit(f"{args.src} 无 offsets 键——先重跑 parse_roboflow_coco.py（v43 版）")
    imgs, labs, offs_all = z["imgs"], z["labels"], z["offsets"]
    print(f"源集：{len(imgs)} 帧", flush=True)
    sess = ort.InferenceSession(str(NANO_V3), providers=["CPUExecutionProvider"])
    keep, pass_cells = nano_judge(imgs, labs, sess,
                                  args.min_agree, args.max_miss, args.min_teacher_conf)

    # ---- 2) 贴片库 ----
    lib = build_patch_library(imgs, labs, offs_all, pass_cells, args.radius)
    if not lib:
        raise SystemExit("贴片库为空：检查机剔阈值与源集")
    # ---- 2b) 原图裁片库合并（v44 VN / v46 zfk+efbok：近景图绕过整帧管线的解锁增量，
    #      开局标准位换子自证 QC≥0.65——augment_*_patch.py 系列产出；v46 起逗号分隔多库）----
    for lib_path in [s.strip() for s in args.extra_lib.split(",") if s.strip()]:
        if not Path(lib_path).exists():
            print(f"额外贴片库缺文件，跳过：{lib_path}", flush=True)
            continue
        zv = np.load(lib_path, allow_pickle=False)
        n_extra = 0
        for k in zv.files:
            if k.startswith("cls"):
                ci = int(k[3:])
                if CLS[ci] in BAN_CLASSES:
                    continue
                lib.setdefault(ci, []).extend(zv[k])
                n_extra += len(zv[k])
        print(f"额外贴片库合并 {Path(lib_path).name}：+{n_extra} 枚 → 总库 "
              f"{sum(len(v) for v in lib.values())}", flush=True)
    classes = sorted(c for c in lib if CLS[c] not in BAN_CLASSES)
    del imgs, offs_all          # 源整帧用完即释（585MB 级）

    # ---- 3)~5) 渲染→贴→分片落盘（流式，库端已质检无需渲染帧 QC）----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(BASE_SEED)

    chunks = [(lo, min(lo + 500, args.frames)) for lo in range(0, args.frames, 500)]
    kept_imgs: list[np.ndarray] = []
    kept_labs: list[np.ndarray] = []
    n_paste_total = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        done = 0
        for chunk in pool.imap(_render_chunk, chunks):
            for i, cls_img, labels in chunk:
                bgr = cv2.cvtColor(cls_img, cv2.COLOR_RGB2BGR)
                lab = labels.astype(np.int64).copy()
                n_paste = rng.randint(max(2, args.per_frame - 1), args.per_frame + 1)
                n_rep = 0 if args.mode == "sparse" else max(
                    1, rng.randint(args.replace_n - 1, args.replace_n + 1))
                plan = pick_paste_plan(lab, rng, classes, n_paste, n_rep)
                for c, ci in plan:
                    patch = rng.choice(lib[ci])
                    cx, cy = AT.cls_cell_center(c // 9, c % 9)
                    AT.paste_piece(bgr, patch, cx, cy)
                    lab[c] = ci
                n_paste_total += len(plan)
                kept_imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                kept_labs.append(lab)
                done += 1
            if done % 2000 < len(chunk):
                print(f"  渲染+贴 {done}/{args.frames}，贴片 {n_paste_total}，"
                      f"{time.time() - t0:.0f}s", flush=True)

    # ---- 分片落盘 ----
    m_total = len(kept_imgs)
    shards = []
    for s in range(0, m_total, args.shard_n):
        p = out_dir / f"{args.tag}_shard{s // args.shard_n}.npz"
        np.savez_compressed(p, imgs=np.stack(kept_imgs[s:s + args.shard_n]),
                            labels=np.stack(kept_labs[s:s + args.shard_n]).astype(np.int64),
                            n_frames=len(kept_imgs[s:s + args.shard_n]))
        shards.append(str(p))
        print(f"落盘 {p}（{p.stat().st_size / 1e6:.0f}MB）")
    meta = {
        "src": args.src, "frames_rendered": args.frames, "keep_frames": m_total,
        "pastes": n_paste_total,
        "lib": {CLS[i]: len(v) for i, v in sorted(lib.items())},
        "cull": {"min_agree": args.min_agree, "max_miss": args.max_miss,
                 "min_teacher_conf": args.min_teacher_conf},
        "shards": shards, "radius": args.radius, "base_seed": BASE_SEED,
        "qc_semantics": "原帧自证（teacher=nano 一次推理同时出帧级机剔门与格级自证门）；"
                        "渲染帧不 QC（先验冲突是特性：贴片教的就是打破格位先验）",
    }
    mp = out_dir / f"{args.tag}_meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== 完成 ==\n产出 {m_total} 帧 / 贴片 {n_paste_total} "
          f"（{sum(Path(p).stat().st_size for p in shards) / 1e6:.0f}MB），"
          f"耗时 {time.time() - t0:.0f}s\nmeta → {mp}")


if __name__ == "__main__":
    main()
