# -*- coding: utf-8 -*-
"""导出与量化（自研识别引擎一期，docs/自研识别引擎方针.md 阶段2）。

链路：best.pt → ONNX fp32 (opset 17) → onnxsim 简化（折叠 BN）→ ir_version 压回
（onnxruntime-android 1.17.1 兼容，先例 quantize_models.py）→ QDQ 静态量化
（校准集=渲染 warp 图；端上无 ConvInteger，动态量化不可用）→ 桌面 ORT 对拍 + 算子审计。

算子白名单（方针 §二）：Conv / MatMul / Gemm / Relu / Pad / Transpose / Reshape /
Add / Mul / BatchNormalization(应被 onnxsim 折叠) / Constant 等——审计时报告全集，
出现黑名单算子（如 Einsum/GridSample）即失败，保证端上 ORT-android 1.17.1 可加载。

产物（runs/<name>/）：gridhead_fp32.onnx、gridhead_int8.onnx、export_report.json
"""
from __future__ import annotations

import os
# 必须在 import torch 之前：export 全程只需 CPU（ckpt 载入 map_location="cpu"），
# 不禁用时 torch import 即初始化 CUDA 上下文（RTX 运行时吃系统 RAM ~1GB+）——
# 2026-10-18 内存事故元凶之一，本进程已列入低内存环境观察名单
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import collections
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

import renderer as R
from model import GridHead, NCLASS, ROWS, COLS, CLS_IN_H, CLS_IN_W

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def str2bool(v: str) -> bool:
    """pipeline 透传布尔参数为 `--attn True` 字符串形式（与 train.py 同款解析）"""
    return str(v).strip().lower() in ("1", "true", "yes", "y")

WHITELIST = {"Conv", "MatMul", "Gemm", "Relu", "Pad", "Transpose", "Reshape", "Add", "Mul",
             "BatchNormalization", "Constant", "Squeeze", "Unsqueeze", "Identity", "Cast",
             # v38a 扩：Softmax——格间注意力的 90×90 归一化（模型侧已用 ReZero 替代 LN、
             # ReLU 替代 GELU 把新增算子压到仅此一个；ORT-android 1.17.1 完整版 CPU EP 原生支持）
             "Softmax",
             # QDQ 静态量化的标准算子（ORT-android 原生支持，内部组合为 QLinearConv；
             # 真正的端上风险算子是动态量化的 ConvInteger——QDQ 路线不会出现）
             "QuantizeLinear", "DequantizeLinear"}

# 16 类记号（与 augment_real_transplant.AT.CLS 一致，分歧解剖用）
CLS_CHARS = ".xKABNRCPkabnrcp"
AT_CLS_NAME = {i: c for i, c in enumerate(CLS_CHARS)}
DATA = Path(__file__).parent / "data"

def normalize_batch(imgs_u8: np.ndarray) -> np.ndarray:
    """uint8 [N,315,280,3] RGB → float32 [N,3,315,280]（与端上 normalizeChwInto 等价）"""
    x = imgs_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def _load_real_frames(paths: list[Path], max_n: int, seed: int) -> np.ndarray:
    """从真实帧池 npz 采样 uint8 [N,315,280,3] RGB（imgs 键），无则返回空。
    v44 修：旧版在第一个存在的 npz 就 return——--calib-real-npz 逗号列表实际
    永远只吃到单源（侦察实测：fallback 的 real_chessai_piece 从未被用到）。
    改为多源均衡采样：每源取 max_n/源数，最后统一截到 max_n。"""
    rng = random.Random(seed)
    pools = []
    for p in paths:
        if not p.exists():
            continue
        z = np.load(p)
        if "imgs" not in z:
            continue
        imgs = z["imgs"]
        print(f"  真实帧来源 {p.name}: 池 {len(imgs)} 帧")
        pools.append(imgs)
    if not pools:
        return np.zeros((0,), np.uint8)
    per = max(1, max_n // len(pools))
    picked = []
    for imgs in pools:
        idx = rng.sample(range(len(imgs)), min(per, len(imgs)))
        picked.append(imgs[idx])
    out = np.concatenate(picked) if len(picked) > 1 else picked[0]
    if len(out) > max_n:
        out = out[rng.sample(range(len(out)), max_n)]
    print(f"  真实帧合计取 {len(out)} 帧（{len(pools)} 源均衡）")
    return out

def calib_reader(n: int = 200, seed: int = 555, real_ratio: float = 0.5,
                 real_npz: list[Path] | None = None):
    """校准数据读取器：渲染 warp 图 + 真实域帧混合。

    v34 教训：纯渲染校准 → int8 掉分集中在真实域（photo_human occ -5.6pt 而渲染域
    几乎不掉），一致率 91.06% 偏低。v35 起默认混合真实帧（渲染域激活范围与真实域
    明显不同，校准集必须覆盖部署时真正会遇到的激活分布）。

    内存优化（2026-10-18，另一元凶）：帧以 u8 常驻（float 的 1/4 体积），
    CalibrationDataReader 惰性逐帧 normalize——float 峰值从整集 n×1.06MB
    （600 帧=640MB 常驻）降到单帧 1MB。喂入顺序与数值与旧实现逐位一致。"""
    import onnxruntime.quantization as Q
    n_real = int(round(n * real_ratio)) if real_ratio > 0 else 0
    n_rend = n - n_real
    fonts = R.load_font_pool()
    rng = random.Random(seed)
    u8_frames: list[np.ndarray] = []
    for _ in range(n_rend):
        s = R.make_sample(rng, fonts)
        u8_frames.append(s["cls_img"])                       # u8 HWC RGB，不预 normalize
    if n_real:
        real = _load_real_frames(real_npz or [], n_real, seed + 1)
        if len(real):
            u8_frames.extend(real[:n_real])
        else:
            print("  [警告] 无可用真实帧池，校准集退化为纯渲染")
    print(f"校准集: {len(u8_frames)} 帧（渲染 {n_rend} + 真实 {len(u8_frames)-n_rend}，"
          f"u8 常驻 {sum(f.nbytes for f in u8_frames)/2**20:.0f}MB 惰性喂入）")

    class DR(Q.CalibrationDataReader):
        """惰性 reader：get_next 逐帧 u8→float32 normalize，float 峰值=单帧"""
        def __init__(self, frames):
            self.frames = frames
            self.it = None
        def get_next(self):
            if self.it is None:
                def gen():
                    for f in self.frames:
                        yield {"input": normalize_batch(f[None])}
                self.it = gen()
            try:
                return next(self.it)
            except StopIteration:
                return None
    return DR(u8_frames)

def agree_stats(sess_fp, sess_q, imgs_u8: np.ndarray, tag: str) -> dict:
    """fp32 vs int8 逐格 argmax 对拍：总体一致率 + 按 fp 预测类分解分歧"""
    agree = cells = 0
    dis_by_cls = collections.Counter()
    dis_pairs = collections.Counter()
    for i in range(len(imgs_u8)):
        x = normalize_batch(imgs_u8[i:i+1])
        lo_f = sess_fp.run(None, {"input": x})[0]
        lo_q = sess_q.run(None, {"input": x})[0]
        af, aq = lo_f.argmax(2), lo_q.argmax(2)
        dis = af != aq
        agree += int((~dis).sum()); cells += af.size
        for c in af[dis].ravel():
            dis_by_cls[int(c)] += 1
        for c, d in zip(af[dis].ravel(), aq[dis].ravel()):
            dis_pairs[(int(c), int(d))] += 1
    top_pairs = {f"{a}->{b}": n for (a, b), n in dis_pairs.most_common(8)}
    stat = {"frames": len(imgs_u8), "cells": cells, "agree": agree / max(1, cells),
            "dis_per_fp_cls": {AT_CLS_NAME[k]: v for k, v in
                               sorted(dis_by_cls.items(), key=lambda kv: -kv[1])},
            "top_confusions": top_pairs}
    print(f"  [{tag}] 一致率 {agree}/{cells} = {stat['agree']:.4f}；分歧最多类 "
          f"{list(stat['dis_per_fp_cls'].items())[:4]}")
    return stat

def audit_ops(path: Path) -> set:
    m = onnx.load(str(path))
    ops = sorted({n.op_type for n in m.graph.node})
    bad = set(ops) - WHITELIST
    print(f"  算子审计: {ops}")
    if bad:
        raise RuntimeError(f"黑名单算子出现: {bad}（端上兼容风险，需改模型或白名单评审）")
    return set(ops)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "runs/v1/best.pt"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--calib-n", type=int, default=50)
    ap.add_argument("--skip-quant", action="store_true")
    ap.add_argument("--width", default="", help="与训练一致的四元组（如 112,168,256,352）；空=默认 64,96,160,224")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--attn", type=str2bool, default=False,
                    help="与训练一致的格间注意力开关（构造模型用，导出图不含 aux 头）")
    ap.add_argument("--xattn", type=str2bool, default=False,
                    help="v62 cross-attention 解码层（Q=格token，K/V=全特征图；ReZero α=β=0 起步）")
    ap.add_argument("--aux-head", type=str2bool, default=False,
                    help="ckpt 若含 aux_fc 权重需开此开关以匹配 state_dict（导出图仅主输出）")
    ap.add_argument("--calib-real", type=float, default=0.5,
                    help="校准集中真实帧占比（v35 起 0.5 默认；0=纯渲染旧口径）")
    ap.add_argument("--calib-real-npz", default="",
                    help="真实帧池 npz（逗号分隔）；空=自动优先 real_rf_piece/real_chessai_piece")
    ap.add_argument("--calib-method", default="Percentile", choices=["MinMax", "Entropy", "Percentile"],
                    help="校准方法（ORT quantize_static calib_method）。v45 起默认 Percentile："
                         "分位数截断抗激活离群帧，50 帧小校准集下 int8 四轴全面反超 fp32 本尊"
                         "（bench 7/10=fp32、style/piece/angle/roboflow +0.3~1.5pt、一致率 0.983/0.990）"
                         "——部署口径修复达成（2026-10-18）。"
                         "⚠️ 收集型方法内存∝帧数：600 帧实爆 31.7GB（每层存全部帧激活分布），"
                         "50 帧水位无感——calib-n 必须小值（≤100）。"
                         "v35 Entropy 崩盘同根因（当时未控帧数维度）。")
    ap.add_argument("--act-sym", action="store_true",
                    help="ActivationSymmetric=True（QUInt8 激活对称化，对照实验用）")
    ap.add_argument("--reduce-range", action="store_true",
                    help="reduce_range=True（int7 量化，治宽模型累加器溢出）")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_dir = Path(args.out_dir or ckpt.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"ckpt": str(ckpt)}

    # 1) 加载模型
    # 加宽配置须与训练一致，否则 load_state_dict 形状不匹配
    width = tuple(int(x) for x in args.width.split(",")) if args.width else (64, 96, 160, 224)
    model = GridHead(width=width, head_dim=args.head_dim,
                     attn=args.attn, aux_head=args.aux_head, xattn=args.xattn)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    report["width"] = list(width)
    report["head_dim"] = args.head_dim
    print(f"加载 {ckpt}（step {ck.get('step')}，指标 {ck.get('metrics', {}).get('in', {})})")

    # 2) 导出 fp32 ONNX（固定 batch=1，与端上一致）
    fp32_path = out_dir / "gridhead_fp32.onnx"
    dummy = torch.randn(1, 3, CLS_IN_H, CLS_IN_W)
    torch.onnx.export(model, dummy, str(fp32_path), input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False,
                      do_constant_folding=True)
    print(f"fp32 导出: {fp32_path} ({fp32_path.stat().st_size/1e6:.2f}MB)")

    # 3) onnxsim 简化（折叠 BN/常量）
    import onnxsim
    m0 = onnx.load(str(fp32_path))
    m1, ok = onnxsim.simplify(m0)
    assert ok, "onnxsim 失败"
    onnx.save(m1, str(fp32_path))
    if m1.ir_version > 8:
        m1.ir_version = 8
        onnx.save(m1, str(fp32_path))
    print(f"onnxsim 完成，ir_version={onnx.load(str(fp32_path)).ir_version}")
    report["fp32_ops"] = sorted(audit_ops(fp32_path))

    # 4) 桌面 ORT fp32 基线延迟
    sess_fp = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fonts = R.load_font_pool()
    rng = random.Random(31337)
    test_imgs = np.stack([R.make_sample(rng, fonts)["cls_img"] for _ in range(60)])
    test_in = normalize_batch(test_imgs)
    for _ in range(5):
        sess_fp.run(None, {"input": test_in[:1]})
    t0 = time.time()
    for _ in range(30):
        sess_fp.run(None, {"input": test_in[:1]})
    fp32_ms = (time.time() - t0) / 30 * 1000
    print(f"fp32 单帧（桌面 CPU 1 线程口径）: {fp32_ms:.1f}ms")
    report["fp32_ms_desktop"] = fp32_ms

    if args.skip_quant:
        json.dump(report, open(out_dir / "export_report.json", "w"), ensure_ascii=False, indent=2)
        return

    # 5) QDQ 静态量化（Conv+Gemm；采样 MatMul 保留 fp32——它是固定常数插值，量化无益）
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
    int8_path = out_dir / "gridhead_int8.onnx"
    real_npz = ([Path(p) for p in args.calib_real_npz.split(",") if p]
                if args.calib_real_npz
                # v45 起三源默认（与 2026-10-18 验证实验逐项一致：渲染+roboflow 实物
                # +App 域贴片——覆盖部署会遇到的全部激活分布，多源均衡采样）
                else [DATA / "real_rf_piece.npz", DATA / "real_photo_chessai.npz",
                      DATA / "real_photo_mix.npz"])
    # v35 注：本机 ORT 1.22 的 CalibrationMethod 是整型枚举，MinMax 为默认值不可显式传
    qkw = {} if args.calib_method == "MinMax" else {
        "calibrate_method": CalibrationMethod[args.calib_method]}
    quantize_static(str(fp32_path), str(int8_path),
                    calib_reader(args.calib_n, real_ratio=args.calib_real,
                                 real_npz=real_npz),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "Gemm"],
                    extra_options=({"ActivationSymmetric": True} if args.act_sym else {})
                                  | ({"reduce_range": True} if args.reduce_range else {}),
                    **qkw)
    mi = onnx.load(str(int8_path))
    if mi.ir_version > 8:
        mi.ir_version = 8
        onnx.save(mi, str(int8_path))
    print(f"int8 导出: {int8_path} ({int8_path.stat().st_size/1e6:.2f}MB)")
    report["int8_ops"] = sorted(audit_ops(int8_path))
    report["fp32_mb"] = fp32_path.stat().st_size / 1e6
    report["int8_mb"] = int8_path.stat().st_size / 1e6

    # 6) int8 对拍：渲染域 + 真实域双口径（v34 教训：纯渲染口径掩盖真实域掉分）
    sess_q = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    stat_render = agree_stats(sess_fp, sess_q, test_imgs, "渲染域")
    report["int8_vs_fp32_agree"] = stat_render["agree"]
    report["int8_disagree_render"] = stat_render
    human26 = DATA / "eval_photo_human.npz"
    if human26.exists():
        z = np.load(human26)
        stat_real = agree_stats(sess_fp, sess_q, z["imgs"][:40], "真实域(human26)")
        report["int8_vs_fp32_agree_real"] = stat_real["agree"]
        report["int8_disagree_real"] = stat_real
    for _ in range(5):
        sess_q.run(None, {"input": test_in[:1]})
    t0 = time.time()
    for _ in range(30):
        sess_q.run(None, {"input": test_in[:1]})
    int8_ms = (time.time() - t0) / 30 * 1000
    print(f"int8 单帧: {int8_ms:.1f}ms（一致率见上双域对拍）")
    report["int8_ms_desktop"] = int8_ms

    # 7) 与旧 nano_v3 同口径延迟对比（同输入、桌面 CPU）
    nano = Path(__file__).resolve().parent / "data/nano_v3-0319.onnx"  # 不可分发：自备可启用对照
    if nano.exists():
        s_nano = ort.InferenceSession(str(nano), providers=["CPUExecutionProvider"])
        iname = s_nano.get_inputs()[0].name
        for _ in range(5):
            s_nano.run(None, {iname: np.ascontiguousarray(test_in[:1])})
        t0 = time.time()
        for _ in range(30):
            s_nano.run(None, {iname: np.ascontiguousarray(test_in[:1])})
        nano_ms = (time.time() - t0) / 30 * 1000
        print(f"nano_v3 单帧（同口径）: {nano_ms:.1f}ms → 新模型加速 {nano_ms/int8_ms:.1f}x")
        report["nano_ms_desktop"] = nano_ms

    json.dump(report, open(out_dir / "export_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"导出报告: {out_dir/'export_report.json'}")

if __name__ == "__main__":
    main()
