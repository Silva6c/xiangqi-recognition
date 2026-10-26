# Changelog

本项目的全部重要变更记录于此。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；版本号语义化（主.次.修）。
研发期逐轮细节（含全部 FAIL 判决）见 [docs/](docs/README.md) 方案链与训练工作台账。

## [Unreleased]

### 待办
- android 参考实现 package 重命名（`com.cabinet.assist` → 中性包名，12 处文档路径联动）
- r3 系判据对称化（presets gates → judges/ 统一格式 + r3_accept.py）

## [1.0.0] - 2026-10-27

首个独立发布：识别引擎从原产品项目分家，成为本仓库的完整开源形态。

### 权重（weights/，fp32 ONNX 三件套）
- `gridhead_fp32.onnx`（11.7MB）—— GridHead 整板棋子识别（r3_w085），90 格 15 类
- `corner_simcc_v1.onnx`（2.5MB）—— CornerSimCC 角点定位（**v4 权重现役**）
- `cell_cls_fp32.onnx`（0.6MB）—— cell_cls 格级精分类（级联仲裁第二级）
- 评测成绩与局限见 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

### 代码
- 训练/评测管线 55 个 Python 模块：corner 系全套（v68 混合头 14 分布契约）、
  r3 系 pipeline + 配方库 `presets/`（v35~v66）、渲染数据工厂、五套评测考卷
- 路径体系统一 `data/` 约定（训练池/评测底片/报告；重建指引 data/README.md）
- `corner_accept.py` 验收自动化：七判据一条命令出 VERDICT（judges/ 判据注册表驱动）
- `tests/` 最小回归 6 项（模型契约/box 往返/残差限幅/体积预算）
- android/ 端上推理核心参考实现（含 boxDecode 混合头契约 + Kotlin 单测）

### 协作体系（多 agent 可接手）
- AGENTS.md 冷启动入口 + 训练协作手册 v2（路由器/任务卡模板/子代理规范）
- 判据唯一可信源 JSON 化（预注册禁改）
- 训练工作台账（<80 行反膨胀，git 历史即归档）

### 文档
- 方案链 v38~v67 全量入档（含全部 FAIL 判决）、训练方法史 v1~v34、模型卡
- 贡献入口 CONTRIBUTING.md + issue/PR 模板

### 已知局限
- 街拍俯拍域（40~50° 大透视）是当前识别链最明确的短板（photo 一致率 ~33%）
- 训练池/评测底片不随仓库分发（来源见 ATTRIBUTION.md 与 data/README.md）
- android/ 为参考实现（无构建工程），集成方按 README 接入
