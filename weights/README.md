# 权重（fp32 ONNX）

现役部署三件套，与 `android/` 端上识别核心配套。部署口径=fp32（int8 量化路线已评估
并放弃：端上 int8 口径错误数为 fp32 的 3 倍，见根文档《我们来时的路》）。

| 文件 | 模型 | 体积 | md5 | 输入 | 输出 |
|---|---|---|---|---|---|
| `gridhead_fp32.onnx` | GridHead 整板识别（r3_w085） | 11.7MB | e4746ecc171a11bd8bf7f51211e14c2b | [1,3,315,280]（RGB，ImageNet 归一化） | [1,90,15] logits |
| `corner_simcc_v1.onnx` | CornerSimCC 角点定位（**v6 权重**，文件名沿用部署名 v1） | 2.5MB | 7ea38479933a699b269684c2263f3645 | [1,3,256,256]（RGB，ImageNet 归一化） | x/y 两个 [1,4,512] 分布（按序取 [0]=x [1]=y） |
| `cell_cls_fp32.onnx` | cell_cls 格级精分类 | 0.6MB | 48d9e7a2c154335365cf757bd900d102 | [90,3,48,48]（RGB /255） | [90,15] logits |

## 推理管线要点

1. **角点**：整帧 → 长边方形窗（短边 pad）+ 1.25 padding affine → 256×256 → SimCC
   argmax/512×256 解码四角（TL/TR/BL/BR = A0/A8/J0/J8 **网格交叉点**，非边框）；
   `boxPlausible` 质量门拦不合法输出（宁 null 不错点，触发全图重定位自愈）。
2. **透视校正**：四角 → warp 至 450×500（角点映射到 (50,50)/(400,50)/(50,450)/(400,450)
   → 中心裁 400×450 → 缩 280×315 RGB。
3. **整板**：280×315 → GridHead → [90,15] → argmax + pmax。
4. **级联**：低置信格（pmax<0.90）由 cell_cls 逐格重判采纳 + 冲突采纳 + 子数校验 +
   对换家族修正（详见 docs/技术文档.md §5）。
5. **朝向正则**：帅(2)/将(9) 行位判 STANDARD/FLIPPED，FLIPPED 时 90 格预测 180° 点镜像。

以上几何与常量在 `android/src/com/cabinet/assist/` 与 python 侧 `eval_screen30.py`
逐位同构，可直接对照阅读。
