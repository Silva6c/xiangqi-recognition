# -*- coding: utf-8 -*-
"""泛化评测集对拍(v33 泛化轮基线与后续选点复用;v35 增第四轴 roboflow)。
用法: python gen_eval.py model1.onnx [model2.onnx ...] [--tag 后缀]
输出: 每模型 × 各轴 cell/occ;角度轴额外按五档阶梯出退化曲线;汇总存 JSON。
轴文件缺失时自动跳过该轴不崩(roboflow 尺由 crop_rf_patches.py 落盘后生效)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

RECOG = Path(__file__).resolve().parent  # 本文件位于 phone/tools/recog/，数据与报告相对此目录
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
RUNG_N = 60  # 角度轴每档帧数


def predict(path, imgs):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    preds = []
    for k in range(len(imgs)):
        x = imgs[k].astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - MEAN) / STD
        preds.append(sess.run(None, {iname: np.ascontiguousarray(x[None])})[0][0].argmax(axis=1))
    return np.stack(preds)


def cell_occ(pred, labels):
    valid = labels != -100
    occ = valid & (labels != 0)
    ok = (pred == labels) & valid
    return float(ok[valid].mean()), float(ok[occ].mean()), int((~ok & valid).sum())


def main():
    argv = sys.argv[1:]
    tag = ""
    out_override = ""
    while "--tag" in argv or "--json" in argv:
        if "--tag" in argv:
            i = argv.index("--tag")
            tag = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
        if "--json" in argv:
            i = argv.index("--json")
            out_override = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    args = argv
    sets = {}
    for name in ("style", "angle", "piece", "roboflow"):
        p = RECOG / "data" / f"eval_gen_{name}.npz"
        if not p.exists():
            print(f"[跳过] 评测轴 {name} 缺失（{p.name} 不存在）")
            continue
        z = np.load(p, allow_pickle=False)
        sets[name] = (z["imgs"], z["labels"])
    report = {}
    for p in args:
        name = Path(p).parent.name + "/" + Path(p).name
        preds = {k: predict(p, v[0]) for k, v in sets.items()}
        row = {}
        for k, (im, lab) in sets.items():
            c, o, n = cell_occ(preds[k], lab)
            row[k] = {"cell": round(c, 4), "occ": round(o, 4), "err": n}
        # 角度轴五档退化曲线
        curve = []
        for r in range(5):
            sl = slice(r * RUNG_N, (r + 1) * RUNG_N)
            c, o, n = cell_occ(preds["angle"][sl], sets["angle"][1][sl])
            curve.append({"rung": r, "cell": round(c, 4), "occ": round(o, 4), "err": n})
        row["angle_curve"] = curve
        report[name] = row
        print(f"\n[{name}]")
        for k in ("style", "piece", "roboflow"):
            if k not in row:
                continue
            v = row[k]
            print(f"  {k:9s} cell={v['cell']:.4f} occ={v['occ']:.4f} 错格={v['err']}")
        print("  angle 退化曲线: " + "  ".join(
            f"R{c['rung']} cell={c['cell']:.4f}/occ={c['occ']:.4f}" for c in curve))
    out = Path(out_override) if out_override else RECOG / "data" / f"gen_eval_report{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()
