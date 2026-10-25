# -*- coding: utf-8 -*-
"""cell_cls 格级精分类器模型（v63 两级自研引擎第二级）。

用途：主模型（GridHead 315×280 整板，格分辨率 ~31.5px）低置信/仲裁格的二次精判。
输入为单格高清 crop（128×128，源图原生格可达 88~100px），信息量显著高于主模型的
31.5px 格——字形混淆（N/R/P/n/r）是对症主战场（26 帧 v2 的 11 错格中 6 格）。

设计约束：小（<1ms/格 CPU）、快训（10~20 分钟/轮）、纯 CNN 保守架构（v59 架构
裁决：小数据强增广下 CNN 不输 transformer）。
"""
import torch
import torch.nn as nn

CELL_IN = 128       # 输入边长
NCLASS = 16         # 与主模型/renderer CLASS_CHARS 一致


class DWBlk(nn.Module):
    """DWConv3×3 → BN → ReLU → PWConv1×1 → BN → ReLU（与 model.py 同构）"""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class CellCls(nn.Module):
    """128×128×3 → 16 类。stem + 5 段 DW（stride2 ×5 → 4×4）+ GAP + fc。"""

    def __init__(self, width=(64, 104, 160, 240, 320)):
        super().__init__()
        chans = [3, *width]
        self.stem = nn.Sequential(
            nn.Conv2d(3, chans[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(chans[1]), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[DWBlk(chans[i], chans[i + 1], 2) for i in range(1, len(width))])
        self.fc = nn.Linear(width[-1], NCLASS)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    m = CellCls()
    n = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n/1e3:.1f}K")
    y = m(torch.randn(2, 3, CELL_IN, CELL_IN))
    print("前向:", y.shape)
