# Third-Party Notices（第三方组件与素材声明）

本仓库分发的代码与权重依赖/参考了以下第三方组件，在此逐一声明。各组件的许可
以上游为准；如本项目许可（GPL-3.0）与组件许可冲突，以更严格者为准。

## 运行时依赖（requirements.txt / android 接入）

| 组件 | 许可 | 用途 |
|---|---|---|
| PyTorch | BSD-3-Clause | 训练/评测 |
| ONNX / ONNX Runtime（含 onnxruntime-android） | MIT | 模型导出与端上推理 |
| OpenCV (opencv-python) | Apache-2.0 | 图像处理 |
| NumPy | BSD-3-Clause | 数值计算 |
| Pillow | MIT-CMU | 渲染数据工厂 |
| TensorFlow Lite（历史 YOLO 兜底对照路径，接入方按需） | Apache-2.0 | android 历史口径 |

## 训练字体（fetch_fonts.py 拉取）

全部为 OFL / 免费商用授权开源字体（55 个， 许可以上游仓库当前版本为准）。
字体本体不随本仓库分发，由脚本按需获取；GEN_HOLDOUT 三个字体仅用于评测隔离。

## 训练/评测数据集

来源与许可逐项见 [ATTRIBUTION.md](ATTRIBUTION.md)（Roboflow Universe 各集
CC BY 4.0、chessai-data、自采与本地未公开数据集）。数据本体不随本仓库分发。

## 参照与上游项目

- [Shirakawa-Kotone/chinese-chess-helper](https://github.com/Shirakawa-Kotone/chinese-chess-helper)：
  项目起点与长期参照系（历史对照口径中的 nano 模型出自其生态，**不随本仓库分发**）。
- [TheOne1006/chinese-chess-recognition](https://github.com/TheOne1006/chinese-chess-recognition)：
  上游对照评测集来源。

## 第三方模型边界（红线）

nano_v3、RTMPose 等**第三方模型权重不随本仓库分发**，也不随任何派生分发物分发；
文档中涉及它们的历史对照数字仅为实验记录。相关重建管线的 teacher 路径已参数化
（环境变量自备）。
