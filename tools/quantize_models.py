# -*- coding: utf-8 -*-
"""ONNX 模型 int8 动态量化脚本(中国象棋识别模型提速用)。

用法:
    python quantize_models.py <fp32模型目录> <输出目录>

原理:权重 int8 / 激活 fp32 的动态量化——CPU 上 x86(AVX-VNNI)与
ARM(dotprod)都有快速路径,Transformer(MatMul 密集)实测分类推理
约 1.5-2.5 倍提速,模型体积约 -70%;无需校准数据集。

只量化 MatMul/Gemm(分类模型的全部计算主体),**不量化 Conv**:
onnxruntime-android(1.17)没有实现 ConvInteger 算子,量化 Conv 会
导致手机端加载即失败(ORT_NOT_IMPLEMENTED,已实测踩坑)。因此
RTMPose(纯 Conv 网)不做量化——它已被角点缓存跳过,只在重建时跑。

转换后把输出目录的 onnx 复制到 phone/app/src/main/assets/models/
覆盖即可(模型文件不进 git)。

注意:onnxruntime-android(1.17)对 IR 版本有上限,新版 onnx 库保存
的模型会带更高 ir_version,脚本里显式压回原模型的 ir_version 保证
手机端能加载。
"""
import os
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

# 只量化分类模型(Swin,MatMul 主体);RTMPose 是 Conv 网,量化在手机端不可用
MODELS = ["nano_v3-0319.onnx"]
QUANT_OP_TYPES = ["MatMul", "Gemm"]


def quantize_one(in_dir: str, out_dir: str, name: str) -> None:
    src = os.path.join(in_dir, name)
    dst = os.path.join(out_dir, name)
    meta = onnx.load(src)
    ir_version = meta.ir_version
    opset = meta.opset_import[0].version
    print(f"{name}: ir_version={ir_version} opset={opset} "
          f"size={os.path.getsize(src) / 1e6:.1f}MB")

    quantize_dynamic(src, dst, weight_type=QuantType.QInt8,
                     op_types_to_quantize=QUANT_OP_TYPES)

    # 压回原 IR 版本,保证 onnxruntime-android 1.17 可加载
    m = onnx.load(dst)
    m.ir_version = ir_version
    onnx.save(m, dst)
    print(f"  -> int8 size={os.path.getsize(dst) / 1e6:.1f}MB (ir_version={m.ir_version})")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for name in MODELS:
        quantize_one(in_dir, out_dir, name)
    print("完成。将输出目录的两个 onnx 复制到 phone/app/src/main/assets/models/ 覆盖。")


if __name__ == "__main__":
    main()
