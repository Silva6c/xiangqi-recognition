# ATTRIBUTION · 训练集来源与许可声明

> 本仓库只发布代码，**不分发任何数据集本体**。本文按许可要求保留全部署名信息。
> 数据仅用于本系统（GridHead 识别模型）的训练与评测。

## 一、Roboflow Universe 公开数据集（逐集核验 CC BY 4.0）

核验方式：Roboflow API `GET /{workspace}/{project}` 的 `project.license` 字段（2026-10-07）。

### 红黑棋子检测集（10 集，约 17.3k 图）

| 来源项目 | 作者/Workspace | 许可 | 图像数 |
|---|---|---|---|
| [xiangqi-super-dataset](https://universe.roboflow.com/312s-workspace/xiangqi-super-dataset) | 312s Workspace | CC BY 4.0 | 4455 |
| [xiangqi-scara-yolo26](https://universe.roboflow.com/312s-workspace/xiangqi-scara-yolo26) | 312s Workspace | CC BY 4.0 | 1734 |
| [xiangqi-standard-pieces](https://universe.roboflow.com/312s-workspace/xiangqi-standard-pieces) | 312s Workspace | CC BY 4.0 | 1037 |
| [scan-xiangqi](https://universe.roboflow.com/bananas-workspace-lcecp/scan-xiangqi) | Bananas Workspace | CC BY 4.0 | 1500 |
| [robot-pi-j9a5f](https://universe.roboflow.com/chess-yyrn7/robot-pi-j9a5f) | chess-yyrn7 | CC BY 4.0 | 3601 |
| [chinese-chess-fpt](https://universe.roboflow.com/finaltest-lqish/chinese-chess-fpt) | FINALTEST | CC BY 4.0 | 1645 |
| [xiangqi-s7nwg](https://universe.roboflow.com/hhb-kgv8h/xiangqi-s7nwg) | hhb | CC BY 4.0 | 1248 |
| [chess-3j3vb](https://universe.roboflow.com/zouling-iolcl/chess-3j3vb) | zouling | CC BY 4.0 | 690 |
| [xiangqi-c8ogg](https://universe.roboflow.com/ccq/xiangqi-c8ogg) | ccq | CC BY 4.0 | 747 |
| [chess-abplc](https://universe.roboflow.com/mqinsulator-o4udf/chess-abplc) | MQinsulator | CC BY 4.0 | 673 |

### 字类标注检测集（1 集，20.6k 图）

| 来源项目 | 作者/Workspace | 许可 | 图像数 |
|---|---|---|---|
| [chinese-chess-seq68](https://universe.roboflow.com/nt-newbie-nheyt/chinese-chess-seq68) | NT Newbie | CC BY 4.0 | 20640 |

类别为字类（ju/ma/pao/shi/zu/bing）+ 独立 red/black 类；红黑归属由 teacher 双色质检判定。

### 棋子分类集（1 集，7.19k 图）

| 来源项目 | 作者/Workspace | 许可 | 图像数 |
|---|---|---|---|
| [chinese-chess-classification-efbok](https://universe.roboflow.com/test1-tgsdm/chinese-chess-classification-efbok) | test1 | CC BY 4.0 | 7190 |

folder 格式（train/valid/test × 10 类：帅仕相马车炮兵卒将象），整图即棋子贴片。

## 二、chessai-data（Zenodo）

- 来源：[nrl-ai/chessai-data](https://github.com/nrl-ai/chessai-data)
- 引用：Nguyen, Viet-Anh. *ChessAI Data - Data for Chinese chess game position recognition*.
  Version 1.0.0. Zenodo, 2023-11-30. DOI: [10.5281/zenodo.1234](https://doi.org/10.5281/zenodo.1234)

## 三、自采数据

- **街拍棋盘照片**：网络爬取（crawl_photos.py，搜索引擎公开图片），仅训练自用
- **真机录屏帧**：自有设备录屏（棋盘 App 界面），仅训练自用

## 四、本地导入数据集（不公开、不分发）

- 三材质实体棋子检测集（888 图 / 28302 框，亚克力/不锈钢/木质；标注：张福凯/大宝 2021-08，
  labelImg 双标注 YOLO+VOC）——未附许可文件，按私有数据内部使用处理，不入库不分发

## 五、评测参照

- 上游对照评测集（17 局街拍/软件棋盘图）参考
  [TheOne1006/chinese-chess-recognition](https://github.com/TheOne1006/chinese-chess-recognition)
  的 docs/sota 展示集整理真值，仅作本地评测参照，不随本仓库分发

## 六、训练字体

由 fetch_fonts.py 拉取的中文开源字体（思源黑体/宋体、LXGW 文楷、ZCOOL 系、Noto 系等），
均为 SIL OFL 或免费商用授权，各字体许可以其上游仓库为准；字体文件不随本仓库分发。
