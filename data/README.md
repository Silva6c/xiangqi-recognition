# data/ 数据缓存（gitignore——本目录不入 git，由本说明重建/迁移）

训练与评测的数据缓存根。**新环境接手时按本说明逐项落位**；脚本一律以仓库根为
工作目录、以 `data/` 为缓存根。

## 落位状态（2026-10-27）

**本机 data/ 已全量落位并核对**：评测底片+训练池 36.5GB（recog/data 平移）、
roboflow_raw 1.3GB、corner 系池 202MB（data/pools/）、fonts 629MB、
外部底片 external 180MB、QQ 基准图 data/qq_basis/。完整性指纹
[docs/data-pools-sha256.txt](../docs/data-pools-sha256.txt)（141 项全对）。
原工作目录副本已按指纹核对后清除。指纹表条目前缀映射：`.cache/xxx` → 本仓 `data/pools/xxx` 或 `data/xxx`，`data/xxx` → 平铺同名——**本目录是唯一副本，谨慎对待**。
真实帧（real_frames/vid_frames）含录屏画面，仅本机训练自用，严禁分发。

## 目录布局

| 位置 | 内容 | 重建方式 |
|---|---|---|
| `data/*.npz`（平铺） | 评测底片（screen30 考卷/human26/bench/四轴等，`build_eval.py`/`gen_screen30.py` 等落盘此处） | 见下「评测底片」 |
| `data/pools/` | 训练池 npz（corner_v5_pool.npz 等，`split` 字段区分 train/holdout） | 见下「训练池」 |
| `data/reports/` | 评测输出（corner_accept 报告、screen30_report.json、冒烟可视化等） | 运行评测自动产生 |

## 字体池（fonts/，仓库根）

`python fetch_fonts.py` 下载重建。GEN_HOLDOUT 三个字体是泛化铁闸：**只允许出现在
评测、永不入训**（训练脚本自动剔除，勿动名单）。

## 训练池（data/pools/）

| 资产 | 来源 | 可否自动重建 |
|---|---|---|
| `corner_v5_pool.npz`（screen 235+vid 39+photo 1498 清洗版） | RTMPose 伪标签+双闸清洗（`pseudo_label_corner.py` 管线） | **否**——依赖第三方 RTMPose 权重（不可分发）与外部照片底片（ATTRIBUTION.md 所列来源，自行获取） |
| `corner_qq_basis.npz`（QQ 事故截图基准，判据 J1） | `qq_basis_pool.py`（v4 定位+目检，需原截图） | 部分——脚本在，原始截图需自备 |
| r3 系各池（roboflow 真监督/贴片库等） | `parse_roboflow_coco.py`、`augment_*.py` 管线 | 部分——依赖外部数据集（自行下载，出处见 ATTRIBUTION.md） |

**接手动作**：从原持有方拷贝 `data/pools/`（或用自备底片跑伪标签管线重建）。
池文件缺失时 corner_train 会自动归零真实占比纯渲染训练——能跑但真实域会崩
（v67 首轮 0% 判废教训，见台账）。

## 评测底片（data/ 平铺）

| 资产 | 生成 | 说明 |
|---|---|---|
| screen30 考卷 | `gen_screen30.py`（参数 seed 20261025 **封版预注册**，永不改动） | 渲染仿真屏 30 帧，真值渲染自证零标注零泄漏 |
| human26 / bench / 四轴 | `build_eval.py` / `build_real_v16.py` 等构建管线 | 依赖外部底片的为不可自动重建资产，同上接手拷贝 |

**铁律**：`data/` 下评测底片一切内容永不入训；考卷 seed 与判据注册表（`judges/`）绑定，
改动=全部口径作废。

## runs/（训练产物，gitignore）

`corner_train.py` / `pipeline.py` 落盘处（best.pt/log.txt/summary.json/导出 onnx）。
不迁移、不入 git；关键数字抄进 `docs/` 留存（数据文件会丢，文档不丢）。
