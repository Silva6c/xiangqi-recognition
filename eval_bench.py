# -*- coding: utf-8 -*-
"""bench 10 帧快评:全对帧数/总错格/逐帧错格位置（自 .cache/bench_eval.py 转正入库）
用法: python eval_bench.py <onnx路径> [more.onnx ...] [--json 输出.json]
"""
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

def main():
    args = sys.argv[1:]
    out_json = ""
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        args = args[:i] + args[i + 2:]
    root = Path(__file__).resolve().parent
    data = np.load(str(root / "data/eval_bench.npz"))
    imgs, labs = data["imgs"], data["labels"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
    report = {}
    for onnx_path in args:
        tag = Path(onnx_path).parent.name + "/" + Path(onnx_path).name
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        full = 0
        total_err = 0
        frames = []
        for i in range(len(imgs)):
            x = (imgs[i].astype(np.float32) - mean) / std
            x = x.transpose(2, 0, 1)[None]
            pred = sess.run(None, {iname: np.ascontiguousarray(x)})[0][0].argmax(axis=1)
            valid = labs[i] != -100
            errs = np.where((pred != labs[i]) & valid)[0]
            total_err += len(errs)
            frames.append(len(errs))
            if len(errs) == 0:
                full += 1
            else:
                pos = [(int(j // 9), int(j % 9)) for j in errs]
                pairs = [f"{'.xKABNRCPkabnrcp'[labs[i][j]]}>{'.xKABNRCPkabnrcp'[pred[j]]}" for j in errs]
                print(f"帧{i}: 错{len(errs)}格 位置{pos} {pairs}")
        print(f"[{tag}] bench: 全对 {full}/10, 总错格 {total_err}")
        report[tag] = {"full": full, "total_err": total_err, "frames_err": frames}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已存 {out_json}")

if __name__ == "__main__":
    main()
