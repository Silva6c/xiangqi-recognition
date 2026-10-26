# AGENTS.md — xiangqi-recognition 识别引擎（独立项目，agent 入口）

自研中国象棋识别引擎：**GridHead 整板棋子分类 + CornerSimCC 角点定位 + cell_cls 格级联**，
独立开源项目（GPLv3）：**https://github.com/Silva6c/xiangqi-recognition**。

> **主仓地位（2026-10-27 起）**：本仓是识别引擎的**唯一演进主仓**——训练/评测/修复
> 单向发生于此。原产品项目（Cabinet）侧的 `phone/tools/recog` 已**冻结为历史基线**
> （只读，仅供其本地诊断脚本临时使用，不回吃本仓修复）。产品端通过 weights/ 三件套
> 与 android/ 推理核心消费引擎；权重更新随本仓发布节奏同步。手机端推理核心在 `android/`，训练/评测管线
在本仓根目录（Python 脚本平铺，`import` 同目录）。本文件是**任何 agent（ZCode / Codex /
OpenCode / DeepSeek / 子代理）的冷启动入口**——读完即可接手，不依赖任何历史会话。

## 冷启动协议（按序读，禁止盲目探查）

1. **`docs/训练工作台账.md`**（<80 行）—— 现在跑着什么、判据进度、下一手任务卡。
   **开工前先读，收工前必更，绑定 git 提交。**
2. **`docs/训练协作手册.md` §0 路由器**（30 行）—— 你是谁/要干嘛 → 哪个剧本。
   铁律（内存三查/红线/权力边界）在 §1，**点火训练前必读**。
3. 本文件下方「硬事实」—— 目录约定、环境、红线。
4. 按需深查：`docs/训练方法史-v1到v34.md`（方法权威+教训全录）、`docs/README.md`
   （全部文档地图）、`README.md`（人类视角 quickstart）。

## 硬事实（路径 / 环境 / 红线）

### 目录约定（分家后与本仓绑定，勿再引用原工作目录路径）

| 路径 | 内容 | git |
|---|---|---|
| `*.py`（根目录平铺） | 训练/评测/数据工厂脚本（约 60 个，同目录 import） | 跟踪 |
| `presets/` | r3 系配方 JSON（v35~v66，每轮唯一主变量+预注册判据） | 跟踪 |
| `judges/` | **判据注册表**（唯一可信源 JSON；corner_accept.py 自动对表） | 跟踪 |
| `weights/` | 现役三件套 fp32 ONNX（md5 与契约见 weights/README.md） | 跟踪 |
| `android/` | 端上推理核心 Kotlin（参考实现，依赖闭包=Piece.java+onnxruntime） | 跟踪 |
| `docs/` | 全部文档（地图见 docs/README.md） | 跟踪 |
| `runs/` | 训练产物（best.pt/log/summary/导出 onnx） | **ignore** |
| `data/` | 数据缓存根（布局与重建见 data/README.md）：评测底片 npz 平铺根 + `data/pools/`（训练池）+ `data/reports/`（评测输出） | **ignore** |
| `fonts/` | 渲染字体池（`fetch_fonts.py` 重建） | **ignore** |
| `tools/` | 端上延迟基准/量化留档等引擎外围工具 | 跟踪 |
| `testdata/` | 评测真值标注（recog-bench cases.json） | 跟踪 |
| `tests/` | 最小回归测试（`python tests/test_corner_model.py`，秒级） | 跟踪 |

### 环境事实

- Python 脚本**一律在仓库根目录执行**（`cd` 到本仓根）；依赖见 `requirements.txt`，
  GPU 训练需本机 CUDA + torch（训练与评测均为本机单机）。
- 本机 32GB 内存：**内存纪律见协作手册 §1.1（起重进程前三查、单重进程、长跑
  run_in_background）**；训练资源分级（用户拍板）：corner 系可拉满
  （batch 256 × workers 16），其他大模型训线**禁止拉满**（r3 系 batch96 内存墙实锤）。
- `cudnn.benchmark` 永禁；评测 val batch 用 48/96 系。
- 网络环境：GitHub 经用户全局代理访问（git 自动走系统代理，无需单独配置）。

### 红线（碰了全部口径作废）

1. **评测集永不入训**：`data/eval/` 下一切 npz、各训练池 `split=="holdout"` 帧、
   GEN_HOLDOUT 字体、screen30 考卷（seed 20261025）、QQ 事故截图基准。
2. **训练点火权在用户**：用户明说"开始/点火/重训"才能起训练进程；评测/目检/文档
   不需请示。
3. `runs/`、`data/`、`fonts/` 不入 git；**评测数字必须写进 docs 留存**。
4. 外部数据集仅训练自用、不分发、记录出处（ATTRIBUTION.md）。
5. 对话、注释、文档、错误信息一律简体中文。
6. 产品端合规：不自动走子、无点击注入。

## 高频操作（corner 系；r3 系走 pipeline.py --preset，见 README）

```bash
# 冒烟（管线验证，~11min）
python corner_train.py --run-name <名>_smoke --steps 300

# 正式（拉满档 ~25min；等效换算 batch256×600 ≈ 96×1500 ≈ 32×4000）
python corner_train.py --run-name <名> --steps 600 --batch 256 --lr 5e-3 --workers 16 \
  --real-npz data/pools/corner_v5_pool.npz --val-every 100

# 验收一条命令（七判据自动对表，末行 VERDICT）
python corner_accept.py --pt runs/<名>/best.pt --judges judges/corner_v68.json

# 导出部署（对拍 <1e-4 才算过）
python corner_export.py --ckpt runs/<名>/best.pt --out runs/<名>
```

## 收工三件事（每轮）

1. 更新 `docs/训练工作台账.md`（状态/判据快照/下一手）→ 2. 关键数字进 docs 留存 →
3. git 提交（台账必须进 commit）。
