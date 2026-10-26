# -*- coding: utf-8 -*-
"""screen30 v1 评测器：三环分层（定位/识别上限/端到端），参数封版见设计文档 §四。

环 1 定位：RTMPose（pose_corner，与端上逐项对齐）成功率+四角对称距离。
环 2 识别上限：GT 四角 warp→r3 fp32→级联；nano 同帧对照（与官方 26 帧口径同构）。
环 3 端到端：屏幕帧→RTMPose→warp450→中心裁 400×450→缩 280×315→r3→级联
（含 unstable 闸模拟）；判据分层 PASS / PASS-badframe / FAIL-wrong。
注意：环 3 的级联逻辑与 eval_cascade_cell.cascade_cell 同构复制（gate→conflict→
子数→对换），改级联实现时两处同步。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_cascade as EC
from eval_cascade_cell import apply_rule_fix_cell
from cell_eval import crop_cells
from cell_model import CellCls

CLS = EC.CLASSES
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
THR, CTHR, CONF = 0.90, 0.60, (0.92, 0.95)      # v63 封版参数
MAX_CHANGED = 13                                 # 端上改格数闸
SWAP_PAIRS = EC.SWAP_PAIRS                       # ({5,6},{12,13}) 对换家族

R3 = str(HERE / "weights/gridhead_fp32.onnx")  # 默认现役权重（仓内），--r3 覆盖
NANO = str(HERE / "../../app/src/main/assets/models/nano_v3-0319.onnx")

# --r3 <onnx>：换整板识别模型（v66 起新 r3 验收口径；不带 = 现役 r3_w085 历史基线）。
# 模块级解析须在会话创建前生效；eval_nano17 等复用本模块的脚本自动继承。
_r3 = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--r3"), None)
if _r3:
    R3 = _r3

sess_r3 = ort.InferenceSession(R3, providers=["CPUExecutionProvider"])
# v66 国产化：nano 已从 assets 删除——评测对照可选（缺失置 None，用到 nano 的口径
# 在调用处自降级；纯自研链路不受影响）
try:
    sess_nano = ort.InferenceSession(NANO, providers=["CPUExecutionProvider"])
except Exception:                                    # noqa: BLE001 缺文件/加载失败
    sess_nano = None
# cell_cls 加载：历史 torch 权重（runs/cell_v3/best.pt，训练产物不入 git）优先，
# 缺失降级 weights/cell_cls_fp32.onnx（同一权重的发布形态，输出 [90,15] 等价）
model = None
_sess_cell = None
try:
    model = CellCls()
    model.load_state_dict(torch.load(str(HERE / "runs/cell_v3/best.pt"), map_location="cpu"))
    model.eval()
except Exception:                                    # noqa: BLE001 训练产物缺失
    _sess_cell = ort.InferenceSession(str(HERE / "weights/cell_cls_fp32.onnx"),
                                      providers=["CPUExecutionProvider"])

from pose_corner import PoseCornerDetector, warp_frame_to_warp450

# --corner-pt runs/xxx/best.pt 或 .onnx：环1/环3 换自研角点模型（CornerInfer，长边窗
# v64 语义）；不带参数 = RTMPose 旧窗（历史基线口径，57% 首考对照）。
_pt = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--corner-pt"), None)
if _pt:
    from corner_infer import CornerInfer
    _kw = {"onnx_path": Path(_pt)} if _pt.endswith(".onnx") else {"torch_weights": Path(_pt)}
    detector = CornerInfer(**_kw)
else:
    # v66 国产化：RTMPose(4_v6-0301) 已从 assets 删除——历史基线口径缺失置 None，
    # 环1/环3 走本函数 main 的地方需自带 --corner-pt 或自行构造
    try:
        detector = PoseCornerDetector()
    except Exception:                                # noqa: BLE001
        detector = None


def cls_input_280x315(w450_bgr: np.ndarray) -> np.ndarray:
    """warp450 → 中心裁 400×450 → 缩 280×315（端上 classifyCells 同构，RGB uint8）"""
    crop = w450_bgr[25:475, 25:425]                       # (500-450)/2, (450-400)/2
    small = cv2.resize(crop, (280, 315), interpolation=cv2.INTER_LINEAR)
    return small[:, :, ::-1]                              # BGR→RGB


def r3_probs(img280_rgb: np.ndarray):
    x = (img280_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
    o = sess_r3.run(None, {"input": x[None]})[0][0]
    p = np.exp(o - o.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    srt = np.argsort(-p, axis=1)
    return {"top1": p.argmax(1), "pmax": p.max(1), "second": srt[:, 1]}


def nano_pred(img280_rgb: np.ndarray) -> np.ndarray:
    x = (img280_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
    lo = sess_nano.run(None, {sess_nano.get_inputs()[0].name: np.ascontiguousarray(x[None])})[0][0]
    return lo.argmax(axis=1)


def cell_out(img280_rgb: np.ndarray):
    cells, _ = crop_cells(img280_rgb, np.zeros(90, np.int64))
    x = cells.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    if model is not None:
        with torch.no_grad():
            lo = model(torch.from_numpy(x))
        pp = torch.softmax(lo, 1)
        c, cp = pp.max(1)
        return cp.numpy(), c.numpy()
    lo = _sess_cell.run(None, {None: x})[0]           # [90,15] logits（onnx 等价）
    e = np.exp(lo - lo.max(-1, keepdims=True))
    pp = e / e.sum(-1, keepdims=True)
    cp = pp.argmax(1)
    return cp, pp.max(1)


def full_cascade(top1, pmax, second, cp, cc):
    """与 eval_cascade_cell.cascade_cell 同构（gate→conflict→子数→对换），返回 p3 与改动数"""
    def adopt(p):
        for g in np.where(pmax < THR)[0]:
            if cc[g] >= CTHR and top1[g] not in (2, 9):
                p[g] = cp[g]
    def conflict_adopt(p):
        hits = np.where((pmax >= CONF[0]) & (cc >= CONF[1]) & (cp != p)
                        & ~np.isin(p, (2, 9)) & ~np.isin(cp, (2, 9)))[0]
        for g in hits:
            p[g] = cp[g]
        return set(int(g) for g in hits)

    p1 = top1.copy(); adopt(p1); ch1 = conflict_adopt(p1)
    p2 = apply_rule_fix_cell(p1.copy(), second, pmax, cp, cc, CTHR)[0] \
        if int((p1 != 0).sum()) >= 8 else p1.copy()
    p3 = top1.copy(); locked = set()
    swaps = EC.find_swaps(top1, second, pmax, THR)
    for g1, g2 in swaps:
        p3[g1], p3[g2] = p3[g2], p3[g1]
        locked |= {g1, g2}
    for g in np.where(pmax < THR)[0]:
        if g not in locked and cc[g] >= CTHR and top1[g] not in (2, 9):
            p3[g] = cp[g]
    ch3 = conflict_adopt(p3)
    if int((p3 != 0).sum()) >= 8:
        p3 = apply_rule_fix_cell(p3, second, pmax, cp, cc, CTHR)[0]
    changed = int((p3 != top1).sum())
    return p3, changed


def detect_orientation_canonical(pred90: np.ndarray):
    """端上 DetectionBoardMapper 同构：帅/将行位判朝向 → FLIPPED 时 180° 点镜像。
    返回 (canonical grid 10×9, orientation)；无帅将 → (None, None)（端上报坏帧）。"""
    g = pred90.reshape(10, 9)
    red_row = int(np.argwhere(g == 2)[:, 0].max()) if (g == 2).any() else -1
    black_row = int(np.argwhere(g == 9)[:, 0].max()) if (g == 9).any() else -1
    if red_row >= 0 and red_row > 4:
        orient = "STANDARD"
    elif black_row >= 0 and black_row > 4:
        orient = "FLIPPED"
    elif red_row >= 0:
        orient = "FLIPPED"
    elif black_row >= 0:
        orient = "STANDARD"
    else:
        return None, None
    return (g if orient == "STANDARD" else g[::-1, ::-1].copy()), orient


def corner_dist(det, gt):
    """检测角(TL,TR,BL,BR) vs GT(TL,TR,BR,BL)：重排后逐点对称距离均值"""
    gt2 = np.array([gt[0], gt[1], gt[3], gt[2]], np.float32)
    return float(np.linalg.norm(det - gt2, axis=1).mean())


def main():
    z = np.load(str(HERE / "data/screen30_v1.npz"), allow_pickle=True)
    imgs, labels, corners = z["imgs"], z["labels"], z["corners"]
    n = len(imgs)

    # ── 环 1：定位 ──
    dets, dists = [], []
    for k in range(n):
        c = detector.detect_with_check(imgs[k][:, :, ::-1])
        dets.append(c)
        dists.append(corner_dist(c, corners[k]) if c is not None else None)
    ok = [d for d in dists if d is not None]
    print(f"== 环1 定位：成功率 {len(ok)}/{n}，对称距离均值 {np.mean(ok):.1f}px 最大 {np.max(ok):.1f}px ==")

    # ── 环 2：识别上限（GT warp） ──
    l0 = l3 = nano_err = 0
    for k in range(n):
        gt2 = np.array([corners[k][0], corners[k][1], corners[k][3], corners[k][2]], np.float32)
        img280 = cls_input_280x315(warp_frame_to_warp450(imgs[k][:, :, ::-1], gt2))
        o = r3_probs(img280)
        cp, cc = cell_out(img280)
        l0 += EC.err_count(o["top1"], labels[k])
        _, _, l3k, _ = __import__("eval_cascade_cell").cascade_cell(
            {**o, "cell": cp, "cell_conf": cc}, labels[k], THR, CTHR, CONF)
        l3 += l3k
        # nano 对照可缺省（国产化后 assets 已删 nano onnx，sess_nano=None）
        if sess_nano is not None:
            nano_err += int((nano_pred(img280) != labels[k]).sum())
    print(f"== 环2 识别上限（GT warp）：r3 L0={l0} → 级联 L3={l3} | "
          f"nano={'缺省' if sess_nano is None else nano_err} ==")

    # ── 环 3：端到端 ──
    e_l0 = e_l3 = badframe = fail_wrong = 0
    details = []
    for k in range(n):
        lab = labels[k]
        if dets[k] is None:
            badframe += 1
            details.append(f"帧{k:02d} 定位失败→报坏帧")
            continue
        img280 = cls_input_280x315(warp_frame_to_warp450(imgs[k][:, :, ::-1], dets[k]))
        o = r3_probs(img280)
        cp, cc = cell_out(img280)
        p3, changed = full_cascade(o["top1"], o["pmax"], o["second"], cp, cc)
        if changed > MAX_CHANGED:
            badframe += 1
            details.append(f"帧{k:02d} 改格{changed}>13→闸拦保原判")
            continue
        # 端上同构朝向自愈：warp 系 p3 → detectOrientation → canonical（红恒 y=9）
        canon, orient = detect_orientation_canonical(p3)
        if canon is None:
            badframe += 1
            details.append(f"帧{k:02d} 无帅将→报坏帧")
            continue
        e_l0 += EC.err_count(o["top1"], lab)
        errs = [(r, c) for r in range(10) for c in range(9)
                if int(lab[r * 9 + c]) >= 0 and int(canon[r, c]) != int(lab[r * 9 + c])]
        e_l3 += len(errs)
        if errs:
            fail_wrong += len(errs)
            det_s = ", ".join(f"({r},{c})真{CLS[int(lab[r*9+c])]}出{CLS[int(canon[r,c])]}@{o['pmax'][r*9+c]:.2f}"
                              for r, c in errs[:6])
            details.append(f"帧{k:02d} FAIL-wrong×{len(errs)} [{orient}]: {det_s}")
    print(f"== 环3 端到端：r3 L0={e_l0} → 管线后 {e_l3} | 报坏帧 {badframe} | FAIL-wrong {fail_wrong} ==")
    for d in details:
        print("  ", d)

    report = {"env1": {"det_ok": len(ok), "n": n, "dist_mean": float(np.mean(ok)),
                       "dist_max": float(np.max(ok))},
              "env2": {"r3_L0": int(l0), "cascade_L3": int(l3), "nano": int(nano_err)},
              "env3": {"r3_L0": int(e_l0), "pipeline_err": int(e_l3),
                       "badframe": int(badframe), "fail_wrong": int(fail_wrong)},
              "details": details}
    (HERE / "data/reports").mkdir(parents=True, exist_ok=True)
    (HERE / "data/reports/screen30_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已存 data/reports/screen30_report.json")


if __name__ == "__main__":
    main()
