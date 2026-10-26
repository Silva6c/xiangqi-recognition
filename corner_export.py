# -*- coding: utf-8 -*-
"""CornerSimCC ONNX 导出（v64 第 2 波 → v67 box 契约）：fp32 单文件（不量化）。

契约固化（v67）：输出按序 (simcc_x, simcc_y) 各 [b,3,512]——x 头 (cx,ux,vx)、
y 头 (cy,uy,vy)，Kotlin 端 OnnxGeom.boxDecode 按输出顺序取 [0][1] 解码 box6→四角，
导出后与 torch 对拍逐位验证；权重内嵌单文件 + ir_version≤8（端上 ORT 1.17.1，
cell_export 同款纪律）；末尾 CPU 延迟快测（对照 4_v6=11.9ms 单线程）。
【雷区】输出形状与 v64 4 角点模型（[b,4,512]）不同——assets 替换必须与端上
boxDecode 原子上线，旧模型+新代码解码错位。

用法：python corner_export.py --ckpt runs/corner_v67/best.pt --out runs/corner_v67
"""
import argparse
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 导出全程 CPU（export.py 同款纪律）

import numpy as np
import torch

from corner_model import CORNER_IN, CornerSimCC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/corner_v1/best.pt")
    ap.add_argument("--out", default="runs/corner_v1")
    args = ap.parse_args()

    model = CornerSimCC()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, CORNER_IN, CORNER_IN)

    import onnx
    out_path = f"{args.out}/corner_simcc_fp32.onnx"
    torch.onnx.export(model, dummy, out_path, opset_version=17,
                      input_names=["input"], output_names=["simcc_x", "simcc_y"],
                      dynamic_axes={"input": {0: "b"},
                                    "simcc_x": {0: "b"}, "simcc_y": {0: "b"}})
    m = onnx.load(out_path, load_external_data=True)
    onnx.save_model(m, out_path, save_as_external_data=False)  # 权重内嵌单文件
    for junk in Path(args.out).glob("corner_simcc_fp32.onnx.data"):
        junk.unlink()
    m = onnx.load(out_path)
    if m.ir_version > 8:
        m.ir_version = 8
        onnx.save(m, out_path)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"fp32 导出(权重内嵌): {out_path}  算子: {ops}")
    print(f"体积: {os.path.getsize(out_path)/1e6:.2f}MB")

    # torch/onnx 对拍（同随机输入逐位）+ 输出顺序契约验证（x 前 y 后）
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    assert names == ["simcc_x", "simcc_y"], f"输出顺序契约破坏: {names}"
    x = np.random.RandomState(0).randn(2, 3, CORNER_IN, CORNER_IN).astype(np.float32)
    with torch.no_grad():
        tx, ty = model(torch.from_numpy(x))
    ox, oy = sess.run(None, {"input": x})
    d = max(np.abs(tx.numpy() - ox).max(), np.abs(ty.numpy() - oy).max())
    print(f"torch/onnx 对拍 max|Δ|={d:.2e}（{'通过' if d < 1e-4 else 'FAIL'}）")

    # CPU 延迟快测（warmup 5 / 30 跑；4_v6 基线 11.9ms 单线程 4.3ms 默认）
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s1 = ort.InferenceSession(out_path, so, providers=["CPUExecutionProvider"])
    xx = x[:1]
    for _ in range(5):
        s1.run(None, {"input": xx})
    t0 = time.time()
    for _ in range(30):
        s1.run(None, {"input": xx})
    print(f"延迟: 单线程 {(time.time()-t0)/30*1000:.2f}ms | ", end="")
    for _ in range(5):
        sess.run(None, {"input": xx})
    t0 = time.time()
    for _ in range(30):
        sess.run(None, {"input": xx})
    print(f"默认线程 {(time.time()-t0)/30*1000:.2f}ms（4_v6 基线: 11.9 / 4.3）")


if __name__ == "__main__":
    main()
