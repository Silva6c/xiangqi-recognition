# -*- coding: utf-8 -*-
"""cell_cls ONNX 导出（v63 第 3 波端上接线用）：fp32 + int8（Percentile 校准）。

纪律：calib-n ≤100（收集型校准内存 ∝ 帧数）；CUDA 显式禁用（export 同款）。

用法：python cell_export.py --ckpt runs/cell_v1/best.pt --out runs/cell_v1
"""
import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 导出全程 CPU（export.py 同款纪律）

import numpy as np
import random
import torch

import cell_data as CD
from cell_model import CellCls


class _CalibReader:
    """渲染高清格校准集（u8 惰性喂入，逐样本 float 化，防收集型内存峰值）。"""

    def __init__(self, n=96, seed=555):
        rng = random.Random(seed)
        fonts = [f for f in CD.R.load_font_pool() if f.split("\\")[-1].split("/")[-1] not in CD.GEN_HOLDOUT]
        self.fonts, self.rng = fonts, rng
        self.n, self.i = n, 0
        self._buf = []

    def get_next(self):
        if not self._buf:
            if self.i >= self.n:
                return None
            cells = CD.render_board_cells(self.rng, self.fonts)
            self._buf = cells[: max(1, len(cells) // 4)]  # 每盘取 1/4，多样性够用
        pil, _ = self._buf.pop()
        arr = CD.augment(pil, self.rng)
        self.i += 1
        return {"input": arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cell_v1/best.pt")
    ap.add_argument("--out", default="runs/cell_v1")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    model = CellCls()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 128, 128)

    import onnx
    torch.onnx.export(model, dummy, f"{args.out}/cell_cls_fp32.onnx", opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}})
    # 权重内嵌为单文件（torch 对 >2MB 张量默认外置 .data——Android assets 打不进伴生文件，
    # 端上加载必炸；load_external_data=True 读回后 save_as_external_data=False 内嵌）
    m = onnx.load(f"{args.out}/cell_cls_fp32.onnx", load_external_data=True)
    onnx.save_model(m, f"{args.out}/cell_cls_fp32.onnx", save_as_external_data=False)
    for junk in Path(args.out).glob("cell_cls_fp32.onnx.data"):
        junk.unlink()
    m = onnx.load(f"{args.out}/cell_cls_fp32.onnx")
    if m.ir_version > 8:   # 端上 ORT 1.17.1 兼容（export.py 同款先例）
        m.ir_version = 8
        onnx.save(m, f"{args.out}/cell_cls_fp32.onnx")
    onnx.checker.check_model(m)
    print(f"fp32 导出(权重内嵌): {args.out}/cell_cls_fp32.onnx")

    if not args.skip_int8:
        from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                               QuantType, QuantFormat, CalibrationMethod)

        class _Reader(CalibrationDataReader):
            def __init__(self, base):
                self.base = base

            def get_next(self):
                return self.base.get_next()

        # 与 export.py 同款调用（本机 ORT 1.22：CalibrationMethod 整型枚举）
        quantize_static(f"{args.out}/cell_cls_fp32.onnx", f"{args.out}/cell_cls_int8.onnx",
                        _Reader(_CalibReader()),
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8, per_channel=True,
                        op_types_to_quantize=["Conv", "Gemm"],
                        calibrate_method=CalibrationMethod.Percentile)
        print(f"int8 导出: {args.out}/cell_cls_int8.onnx")


if __name__ == "__main__":
    main()
