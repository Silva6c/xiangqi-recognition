# 端上识别核心（Android / Kotlin）

从内阁（Cabinet）App 中抽出的**识别链核心代码**——手机屏上棋盘的实时识别就是这段
代码在跑。以参考实现形式发布：与 `weights/` 三件套配套，可独立接入你自己的 App。

## 文件与职责

| 文件 | 职责 |
|---|---|
| `src/com/cabinet/assist/OnnxBoardRecognizer.kt` | 识别入口：角点模型 → 透视校正 → GridHead → 级联 → 朝向正则 的完整编排（含模型会话管理、A/B 开关） |
| `src/com/cabinet/assist/OnnxGeom.kt` | 几何原语：affine 窗、simccKeypoints 解码（v4 现役 [1,4,512] 契约）、`boxPlausible` 质量门、boxDecode（v67/v68 box 参数化实验线 [1,3,512]→[1,7,512] 契约+结构护栏）、warp450 |
| `src/com/cabinet/assist/CellCascade.kt` | 四通道级联仲裁（低置信门控/冲突采纳/子数校验/对换家族） |
| `src/com/cabinet/assist/DetectionBoardMapper.kt` | 帅/将行位朝向判定 + 180° 点镜像正则化 |
| `src/com/cabinet/assist/RecognitionTypes.kt` | 识别结果数据类型 |
| `src/com/cabinet/assist/BoardGrid.kt` | 90 格网格几何（格心/裁格） |
| `src/com/cabinet/assist/YoloBoardDetector.kt` + `YoloPostprocessor.kt` | 历史口径（第三方 RTMPose/YOLO 角点基线）保留作 A/B 对照 |
| `src/com/cabinet/gamelogic/Piece.java` | 棋子枚举（识别链唯一外部类型依赖） |

## 接入要点

1. **依赖**：`com.microsoft.onnxruntime:onnxruntime-android`；模型放
   `app/src/main/assets/models/`（三件套见 `weights/`）。
2. **推理配置**：fp32 模型建议 `ORT Session` 开启 NNAPI/XNNPACK 视机型实测；本仓库
   训练侧的 CPU 延迟口径见 docs/技术文档.md §8。
3. **失败语义**：`boxPlausible` 拦截角点输出返回 null——调用方应触发全图重定位而非
   采用错误角点（宁 null 不错点）。
4. `YoloBoardDetector` 路径需要第三方 RTMPose 模型（**不随本仓库分发**），仅复现
   历史对照实验时使用；正式链路请使用 CornerSimCC（`corner_simcc_v1.onnx`）。
5. 这些文件原属 `com.cabinet.assist` 包，与悬浮窗 UI、引擎通信等服务层解耦干净
   （依赖闭包仅 `Piece.java` + onnxruntime + Android 框架），改包名即可平移。
