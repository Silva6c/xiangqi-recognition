# -*- coding: utf-8 -*-
"""26 帧人工真值评测集分域对拍（v20 轮起复用，自 .cache/human26_eval.py 转正入库）。

域划分(与快照"17 实物 + 9 App"一致,按来源文件名序):
  0~16  实物照片域(街头/公园/实拍/对弈/棋摊/残局摆棋/视频帧)
  17~25 App/教学图域(残局图/习题/教学棋盘/俯拍高清)

用法:
  python eval_human26.py path/to/gridhead_fp32.onnx [more.onnx ...] [--json 输出.json]
nano 单独跑:python eval_human26.py --nano
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
NANO = str(HERE / "../../app/src/main/assets/models/nano_v3-0319.onnx")
IM_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
SPLIT = 17   # 0~16 实物,17~25 App
DOMAINS = (("实物17", slice(0, SPLIT)), ("App域9", slice(SPLIT, 26)), ("合计26", slice(None)))


def eval_onnx(path: str, tag: str, imgs, labels):
    """返回 {域: {cell/occ/err}} 结果字典，stdout 输出与旧版逐字兼容。"""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    preds = []
    for k in range(len(imgs)):           # 导出图固定 batch=1,逐帧推理
        x = imgs[k].astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - IM_MEAN) / IM_STD
        lo = sess.run(None, {iname: np.ascontiguousarray(x[None])})[0][0]  # [90,16]
        preds.append(lo.argmax(axis=1))
    pred = np.stack(preds)
    valid = labels != -100
    occ = valid & (labels != 0)          # 0='.'(空格)
    ok = (pred == labels) & valid
    result = {}
    for name, sl in DOMAINS:
        v, o, k = valid[sl], occ[sl], ok[sl]
        n_err = int((~k & v).sum())
        result[name] = {"cell": round(float(k[v].mean()), 4),
                        "occ": round(float(k[o].mean()), 4), "err": n_err}
        print(f"  {tag} {name}: cell={k[v].mean():.4f} occ={k[o].mean():.4f} 错格={n_err} "
              f"(判定 {int(v.sum())}/占子 {int(o.sum())})")
    return result


def main():
    args = sys.argv[1:]
    npz = "data/eval_photo_human.npz"
    out_json = ""
    if "--npz" in args:                     # v50 双轨：--npz data/eval_photo_human_v2.npz
        i = args.index("--npz")
        npz = args[i + 1]
        args = args[:i] + args[i + 2:]
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        args = args[:i] + args[i + 2:]
    z = np.load(str(HERE / npz), allow_pickle=False)
    imgs, labels = z["imgs"], z["labels"]
    report = {}
    if args and args[0] == "--nano":
        report["nano_v3"] = eval_onnx(NANO, "nano_v3", imgs, labels)
    for p in args:
        tag = p.replace("\\", "/").split("/")[-2] + "/" + p.replace("\\", "/").split("/")[-1]
        report[tag] = eval_onnx(p, tag, imgs, labels)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已存 {out_json}")


if __name__ == "__main__":
    main()
