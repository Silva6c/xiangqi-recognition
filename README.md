# 中国象棋识别系统（GridHead）

纯自研的中国象棋棋盘/棋子识别系统：从渲染数据工厂、三段式识别模型到分层评测考卷的
完整训练与评测管线。识别链不依赖任何第三方识别模型。

## 系统组成

三段式识别链 + 级联仲裁：

| 环节 | 模型 | 说明 |
|---|---|---|
| ① 角点定位 | **CornerSimCC**（633K / 2.54MB fp32） | 一维分布剖面回归四角（SimCC 式），长边方形窗 + 1.25 padding affine，输出 TL/TR/BL/BR 网格交叉点 |
| ② 整板识别 | **GridHead**（~2.8M / 11.7MB fp32） | 90 格联合分类（280×315 输入），格间注意力（ReZero 残差）+ 占子辅助头 + 格条件类平衡损失 |
| ③ 格级精分类 | **cell_cls**（cell 级小分类器） | 对低置信格逐格重判，四通道级联仲裁：低置信单向门控 / 冲突采纳 / 子数校验 / 对换家族 |

识别类别 `".xKABNRCPkabnrcp"`（空/占位/帅仕相马车炮兵 + 黑方小写），朝向自适应
（帅/将行位判 STANDARD/FLIPPED，180° 点镜像正则化）。

## 目录速览

```
├── corner_model.py / corner_train.py / corner_export.py / corner_infer.py
│     角点模型（CornerSimCC）：训练/导出/推理（含 boxPlausible 质量门）
├── model.py                     整板模型 GridHead（注意力/辅助头/解码层开关）
├── train.py / export.py         整板训练与 fp32/int8 导出
├── cell_model.py / cell_train.py / cell_export.py
│                                 格级精分类器三件套
├── renderer.py                  在线渲染数据工厂（贴片库/光度弹药/遮挡/UI 背景模拟）
├── corner_data.py / cell_data.py  渲染放置几何与增广（角点 GT 同步变换）
├── pipeline.py                  一键训练编排：预检→watchdog→训练→导出→评测→报告
├── presets/                     配方库 v35~v66（每轮唯一主变量 + 预注册判据 gates）
├── eval_human26.py              26 帧人工真值金口径
├── eval_bench.py                10 帧难例考卷
├── eval_gen.py                  四轴泛化（字体/棋子/角度/贴片）
├── eval_screen30.py             screen30 三环考卷（定位/识别上限/端到端）
├── eval_nano17.py               上游对照评测（17 局，full-err-k 口径）
├── eval_cascade*.py             级联仲裁评测
├── augment_*_patch.py           外部数据裁片/贴片管线（roboflow/zfk/efbok/VN）
├── parse_roboflow_coco.py       COCO 框-格映射入池管线
├── make_photo_warp_pool.py      街拍 warp 池工厂（伪标签+编制闸+一致格 IGNORE）
├── fetch_fonts.py               训练字体拉取（SIL OFL 授权字体）
└── ATTRIBUTION.md               训练集来源与许可声明（本仓库不分发数据本体）
```

## 快速开始

```bash
pip install torch onnxruntime opencv-python numpy

# 1) 拉取训练字体（SIL OFL）
python fetch_fonts.py

# 2) 一键训练（配方=presets/v66.json，训练产物落 runs/<名>/）
python pipeline.py --preset v66 --run-name v66 --baseline runs/v60/auto_summary.json

# 3) 评测（金口径/难例/四轴泛化）
python eval_human26.py runs/v66/human_fp32.onnx
python eval_gen.py runs/v66/gen_fp32.onnx
```

> 注意：本仓库按"研究管线"原样开源——数据集本体、训练权重、评测 npz 不随仓库分发
> （来源与获取方式见 [ATTRIBUTION.md](ATTRIBUTION.md)）；部分评测脚本中的模型路径为
> 作者本机绝对路径，使用时按需修改。

## 训练集来源

全部训练数据的来源与许可逐集列于 **[ATTRIBUTION.md](ATTRIBUTION.md)**，概要：

- **Roboflow Universe 公开数据集**（13 集，约 4.5 万图，逐集核验 CC BY 4.0）
- **chessai-data**（nrl-ai，Zenodo DOI 10.5281/zenodo.1234）
- **自采数据**：网络爬取街拍棋盘照片、真机录屏帧（仅训练自用，不分发）
- 另有本地导入数据集（未公开许可，仅内部训练，不入库不分发）

本仓库**只发布代码，不分发任何数据集本体**；衍生权重亦不随仓库分发。

## License

- 代码：**GPL-3.0**（见 [LICENSE](LICENSE)）
- 训练字体：OFL / 免费商用授权开源字体（由 fetch_fonts.py 获取，各字体许可以其上游为准）
- 数据集：各自许可见 ATTRIBUTION.md（CC BY 4.0 等署名要求在该文件中保留）
