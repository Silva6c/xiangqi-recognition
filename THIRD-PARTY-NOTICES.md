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
| fontTools | BSD-3-Clause | 可变字体实例化（fetch_fonts.py 扩池） |
| requests | Apache-2.0 | 街拍数据爬取（crawl_photos.py） |
| TensorFlow Lite（历史 YOLO 兜底对照路径，接入方按需） | Apache-2.0 | android 历史口径 |

## 训练字体（fetch_fonts.py 拉取）

全部为 OFL / 免费商用授权开源字体（38 个开源字体落盘，另含本机系统字体仅本地
评测用不分发），**逐项目作者署名见 [ATTRIBUTION.md](ATTRIBUTION.md) §六**，
许可以上游仓库当前版本为准。字体本体不随本仓库分发，由脚本按需获取；
GEN_HOLDOUT 三个字体仅用于评测隔离。

## 训练/评测数据集

来源与许可逐项见 [ATTRIBUTION.md](ATTRIBUTION.md)（Roboflow Universe 12 集
CC BY 4.0 逐集署名、chessai-data、自采与本地未公开数据集）。数据本体不随本仓库分发。

## 参照与上游项目

- [yolo12138/Chinese_Chess_Recognition](https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition)
  （**MIT**）：历史对照权重 nano_v3-0319 与 4_v6-0301（RTMPose SimCC 角点）的
  **原作者发布仓**——2026-10-27 考古确认 4_v6 与本地副本逐字节一致、nano 为其
  INT8 量化版（契约/argmax 逐位一致）。该空间即上游识别管线的源头。
- [Vincentzyx/VinXiangQi](https://github.com/Vincentzyx/VinXiangQi)（GPL-3.0）：
  历史对照模型与 YOLO 解码策略来源（android 参考实现的 YoloBoardDetector/
  YoloPostprocessor 移植自其 GetBoardFromPrediction，头注释保留出处）；
  其作者 Vincentzyx 亦为 Pikafish 创始作者之一。对照模型不随本仓库分发。
- [TheOne1006/chinese-chess-recognition](https://github.com/TheOne1006/chinese-chess-recognition)：
  上游对照评测集来源（nano17 考卷 17 局与其 sota 对比表口径）。
- [OpenMMLab RTMPose](https://github.com/open-mmlab/mmpose)（Apache-2.0）：
  角点定位架构出处（4_v6 权重为其 SimCC 风格架构），历史对照口径的基线之一。

## 第三方模型边界（红线）

nano_v3、RTMPose 权重等**第三方模型权重不随本仓库分发**，也不随任何派生分发物
分发；文档中涉及它们的历史对照数字仅为实验记录。相关重建管线的 teacher 路径已
参数化（环境变量自备）。权重原作者发布（yolo12138，MIT）见上节——自取权重时
请同时遵守 MIT 的版权与许可声明保留义务。
