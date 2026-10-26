# 中国象棋识别系统（GridHead）

> 仓库：**https://github.com/Silva6c/xiangqi-recognition**

**这是一个完全由 AI 自主训练出来的象棋识别系统。** 从数据工厂、模型架构、训练
配方到评测裁决，全程由 AI 完成：**GLM-5.3** 主力驱动（方案设计/代码实现/训练点火/
评测裁决），**GLM-5.3-Flash** 与 **DeepSeek-V4.1-Flash** 并行协作（子任务分解/
独立视角复核），人类负责提出目标、供给原始数据、拍板关键决策。两个月的 66 轮
配方迭代，每轮判据预注册、结果全部入档——研发全过程的来龙去脉见
[《我们来时的路》](docs/我们来时的路.md)。

识别链分三段：**CornerSimCC 角点定位 → GridHead 90 格整板识别 → cell_cls 格级
精分类 + 级联仲裁**，配五套分层评测考卷。对社区参照模型（nano）在全部口径上
领先：26 帧人工金口径错误 11→2（级联）、四轴泛化 +1~5pt、上游 17 局对照识别环
错 30 vs 49。

**两仓关系**：本仓是识别引擎的独立完整项目（训练管线/评测器/权重/端上推理核心/
文档），从原产品项目（Go 引擎服务 + 悬浮窗 App）分家而来；产品端通过
`weights/` 三件套与本仓 `android/` 推理核心消费引擎，引擎演进按「改动→验收
（judges 判据）→同步权重与 android 核心」的节奏双向同步。

## 模型权重

现役 fp32 三件套已随仓库发布（`weights/`，共 ~14.8MB），输入输出规格与推理管线
见 [weights/README.md](weights/README.md)：

| 文件 | 模型 | 体积 |
|---|---|---|
| `weights/gridhead_fp32.onnx` | GridHead 整板识别 | 11.7MB |
| `weights/corner_simcc_v1.onnx` | CornerSimCC 角点定位（**v4 权重**现役，v6 因 ornate 域塌缩事故回退见 docs/训练工作台账.md） | 2.5MB |
| `weights/cell_cls_fp32.onnx` | cell_cls 格级精分类 | 0.6MB |

## 端上核心代码

手机端实时识别链的核心 Kotlin 实现在 [`android/`](android/README.md)（从生产 App
中抽出，依赖闭包干净：onnxruntime-android + 一个棋子枚举），与 `weights/` 直接配套。

## 目录速览

```
├── AGENTS.md                    agent 冷启动入口（ZCode/Codex/OpenCode/DeepSeek 通用）
├── requirements.txt             Python 依赖（本机验证版本）
├── tests/                       最小回归测试（改模型/解码/几何后必跑，秒级）
├── data/                        数据缓存根（gitignore，重建指引见 data/README.md）
├── weights/                     部署权重三件套（fp32 onnx）+ 规格说明
├── android/                     端上识别核心（Kotlin，生产代码抽出的参考实现）
├── docs/                        全部文档（地图 docs/README.md；方案链 v38~v67 + 协作体系）
├── judges/                      判据注册表 JSON（唯一可信源）
├── tools/ bench_real_device.sh  端上延迟基准/量化留档等外围工具
├── testdata/recog-bench/        评测真值标注（bench_recog 配套）
├── docs/技术文档.md             架构/数据体系/评测方法/部署口径
├── docs/我们来时的路.md         两个月 66 轮的研发叙事（含全部 FAIL）
├── model.py                     GridHead（格间注意力/占子辅助头/解码层开关）
├── corner_model.py 等           CornerSimCC 训练/导出/推理（boxPlausible 质量门）
├── cell_model.py 等             cell_cls 三件套
├── train.py / export.py         整板训练与 fp32 导出
├── renderer.py / corner_data.py 在线渲染数据工厂（角点 GT 解析自证）
├── pipeline.py                  一键训练编排：预检→watchdog→训练→导出→评测→报告
├── presets/                     配方库 v35~v66（每轮唯一主变量 + 预注册判据）
├── judges/                      判据注册表（唯一可信源 JSON，corner_accept.py 自动对表）
├── eval_human26.py 等           五套考卷：26帧金口径/bench/四轴/screen30/nano17
├── augment_*_patch.py           外部数据裁片/贴片管线
├── make_photo_warp_pool.py      半监督伪标签池工厂（一致格 IGNORE）
├── fetch_fonts.py               训练字体拉取（OFL/免费商用授权）
└── ATTRIBUTION.md               训练集来源与许可声明（数据本体不随仓库分发）
```

## 快速开始

```bash
pip install -r requirements.txt   # torch 按 pytorch.org 选对应 CUDA 版

# 0) 数据落位（关键：训练池/评测底片不随仓分发）
#    评测底片部分可自动重建（build_eval.py / gen_screen30.py）；
#    训练池与外部底片来源见 data/README.md 与 ATTRIBUTION.md
python fetch_fonts.py             # 训练字体拉取

# 1) 角点系训练（拉满档 ~25min；训练前必读 docs/训练协作手册.md §1 铁律）
python corner_train.py --run-name <名> --steps 600 --batch 256 --lr 5e-3 --workers 16   --real-npz data/pools/corner_v5_pool.npz --val-every 100

# 2) 一条命令验收（七判据自动对表，末行 VERDICT）
python corner_accept.py --pt runs/<名>/best.pt --judges judges/corner_v68.json

# 3) r3 系一键训练（配方库 presets/v35~v66）
python pipeline.py --preset v66 --run-name v66   # 首跑无上轮产物可省略 --baseline

# 4) 评测五套考卷
python eval_human26.py weights/gridhead_fp32.onnx
```

> 本仓库按研究管线原样开源：数据集本体、评测 npz 不随仓库分发（获取方式见
> [ATTRIBUTION.md](ATTRIBUTION.md) 与 [data/README.md](data/README.md)）；第三方模型
> （nano/RTMPose）不随仓库分发。

## 多 agent 协作（ZCode / Codex / OpenCode / DeepSeek 皆可接手）

本项目的全部状态与协议在文件里，**不依赖任何历史会话记忆**：

- **[AGENTS.md](AGENTS.md)**：任何 agent 框架的冷启动入口——目录约定、红线、
  高频命令（各家 agent 工具均识别根目录 AGENTS.md 约定）
- **[docs/训练协作手册.md](docs/训练协作手册.md)**：协作协议 v2——角色与权力边界、
  内存铁律、四张任务卡模板（训练/验收/部署/事故）、判据体系、子代理调用规范
  与 prompt 模板（必读只有 §0 路由器 30 行）
- **[docs/训练工作台账.md](docs/训练工作台账.md)**：去中心化状态文件（开工先读、
  收工必更、<80 行反膨胀，git 历史即归档）
- **[judges/*.json](judges/)**：判据唯一可信源 + `corner_accept.py` 自动对表——
  评测判读从"人工读百行输出"压缩为"读一行 VERDICT"，杜绝判读漂移

## 技术文档

- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)：**模型卡**——用途/契约/成绩/局限明示/许可立场
- [CHANGELOG.md](CHANGELOG.md)：版本变更记录（v1.0.0 = 现役三件套首发）
- [CONTRIBUTING.md](CONTRIBUTING.md)：贡献指南（代码 PR / 数据贡献格式 / agent 接手回路）
- [docs/技术文档.md](docs/技术文档.md)：三段式管线、SimCC 目标与窗语义、格条件类
  平衡损失、级联仲裁四通道、数据工厂与评测隔离纪律、部署口径（fp32，int8 已弃）。
- [docs/我们来时的路.md](docs/我们来时的路.md)：从渲染工厂第一行代码到系统级反超，
  两个月的完整叙事——包括全部失败轮次。

## 训练集来源

全部训练数据的来源与许可逐集列于 **[ATTRIBUTION.md](ATTRIBUTION.md)**（Roboflow
Universe 13 集 CC BY 4.0、chessai-data Zenodo、自采数据、本地未公开数据集）。
本仓库只发布代码与权重，不分发任何数据集本体。

## 致谢

- [Shirakawa-Kotone/chinese-chess-helper](https://github.com/Shirakawa-Kotone/chinese-chess-helper)
  （CCH）：项目的起点与长期参照系。
- [TheOne1006/chinese-chess-recognition](https://github.com/TheOne1006/chinese-chess-recognition)：
  上游对照评测集来源，让"领先"有了外部锚点。
- **一切数据提供者**：Roboflow Universe 的数据集作者们、chessai-data 的
  Nguyen Viet-Anh、三材质数据集标注者张福凯/大宝，以及所有公开棋盘照片的拍摄者。

## 参与贡献

**欢迎更多人参与进来。** Issues / PR 都开放，特别欢迎这些方向：

- **数据**：更多真实场景棋盘照片/录屏（含标注），尤其是街拍俯拍域——这是当前
  识别链唯一明确的短板
- **评测**：新考卷设计与真值校验（我们的评测方法论：判据预注册+数字入档，欢迎
  挑战与扩展）
- **端上**：NNAPI/GPU delegate 实测与移植、更多机型的延迟数据
- **训练**：渲染弹药扩展、新配方实验（presets 体系加一个 JSON 就能开新轮）
- **协作**：用你惯用的 agent 框架按 AGENTS.md + 协作手册接手一轮训练/评测，
  压测这套去中心化协作流程本身

## License

- 代码：**GPL-3.0**（见 [LICENSE](LICENSE)）
- 权重：随代码同许可分发；训练数据各自许可见 ATTRIBUTION.md
- 训练字体：OFL / 免费商用授权开源字体（由 fetch_fonts.py 获取，许以上游为准）
