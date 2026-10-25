# -*- coding: utf-8 -*-
"""合成棋盘渲染器（自研识别引擎的数据引擎核心，docs/自研识别引擎方针.md 阶段1）。

职责：零人工标注地生成与端上识别管线分布对齐的训练/评测样本。

与端上契约的对齐（逐项核实自 OnnxBoardRecognizer.kt / OnnxGeom.kt）：
- warp 空间 450×500，棋盘网格四角映射到 (50,50)/(400,50)/(50,450)/(400,450)；
- 网格 9 列 10 行：列距 43.75px、行距 44.444px（warp 空间）；
- 分类输入 = warp 后中心裁 400×450（裁掉每边 25px）→ 缩放 280×315（×0.7）→ ImageNet RGB 归一化
 （归一化在训练循环做，这里只出 uint8）；
- 16 类顺序与 OnnxGeom.kt:18 的 CLASS_CHARS 逐字符一致。

设计要点：
- 物理级增广在渲染期完成（主题/字体/透视/光照/遮挡），训练循环只做张量归一化；
- 「整屏帧」模式同时产出 4 角标注（为二期角点模型免费储备）；
- 角点抖动（模拟 RTMPose 定位误差）在 warp 回采样时施加，让分类头对几何误差鲁棒；
- 遮挡物覆盖的格子 label=-100（训练时忽略），与「不确定就闭嘴」的失败哲学一致。

许可证红线：本文件为全新自研代码，未复制任何上游实现（见方针 §七）。
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ==================== 与端上管线对齐的常量（勿改，改动=契约破坏） ====================
WARP_W, WARP_H, WARP_PAD = 450, 500, 50          # OnnxBoardRecognizer WARP_*
CLS_CROP_W, CLS_CROP_H = 400, 450                # CLS_CROP_*
CLS_IN_W, CLS_IN_H = 280, 315                    # CLS_IN_*（W×H）
COLS, ROWS = 9, 10
# 16 类顺序，与 OnnxGeom.kt:18 CLASS_CHARS 逐字符一致
CLASS_CHARS = ".xKABNRCPkabnrcp"
CLASS_INDEX = {ch: i for i, ch in enumerate(CLASS_CHARS)}
IGNORE_LABEL = -100

def col_x(c: int) -> float:
    """warp 空间第 c 列交叉点 x"""
    return WARP_PAD + (WARP_W - 2 * WARP_PAD) * c / (COLS - 1)

def row_y(r: int) -> float:
    """warp 空间第 r 行交叉点 y"""
    return WARP_PAD + (WARP_H - 2 * WARP_PAD) * r / (ROWS - 1)

# 棋子文字（红大写/黑小写，与 FEN 字母一一对应）
GLYPHS = {"K": "帅", "A": "仕", "B": "相", "N": "马", "R": "车", "C": "炮", "P": "兵",
          "k": "将", "a": "士", "b": "象", "n": "马", "r": "车", "c": "炮", "p": "卒"}

# 字形变体池（v18）：真实 App/实物棋子红方常用繁体（帥傌俥砲車馬），黑方也有繁体。
# 旧版红方只渲染简体 → "繁体红子"从未进过训练（bench cnvcs 底排 車馬帥 实测、
# App 补爬池底排整行把红子认成黑子的根因），模型只能把繁体红子映射到用繁体
# 字形的黑方类。每子独立随机选变体，字形本身不再是颜色判据。
GLYPH_POOLS = {
    "K": ("帅", "帥"), "A": ("仕",), "B": ("相",),
    "N": ("马", "馬", "傌"), "R": ("车", "車", "俥"), "C": ("炮", "砲", "包"), "P": ("兵",),
    "k": ("将", "將"), "a": ("士",), "b": ("象",),
    "n": ("马", "馬"), "r": ("车", "車"), "c": ("炮", "砲", "包"), "p": ("卒",),
}


def glyph_of(ch: str, rng: random.Random, fpath: str | None = None) -> str:
    """按类从字形变体池随机取一个字形；fpath 给定时按该字体覆盖过滤，
    防止生僻字（傌/俥）在缺字字体上画出空盘（空白子会教模型对空格乱猜子）。"""
    pool = GLYPH_POOLS.get(ch)
    if not pool:
        return GLYPHS.get(ch, ch)
    if fpath is not None:
        cov = _font_covered(fpath)
        hit = [g for g in pool if g in cov]
        if hit:
            return rng.choice(hit)
        base = GLYPHS.get(ch, pool[0])
        return base if base in cov else pool[0]
    return rng.choice(pool)


_FONT_COVER_CACHE: dict = {}


def _font_covered(fpath: str) -> set:
    """字体 cmap 覆盖字符集（每字体只读一次）"""
    s = _FONT_COVER_CACHE.get(fpath)
    if s is None:
        try:
            from fontTools.ttLib import TTFont
            f = TTFont(fpath, fontNumber=0, lazy=True)
            s = {chr(c) for c in f.getBestCmap()}
            f.close()
        except Exception:
            s = set()
        _FONT_COVER_CACHE[fpath] = s
    return s


# 墨色解耦（v18）：cnvcs 红方用蓝墨、部分 App 黑方用绿/蓝墨——墨色不能是红黑
# 类的唯一判据，字形+盘面样式要共同决定。红/黑各留概率走替代墨色池。
RED_INK_ALT = ((30, 60, 200), (25, 45, 170), (40, 100, 70), (70, 70, 90))
BLACK_INK_ALT = ((30, 110, 60), (35, 70, 190), (140, 50, 50), (70, 70, 90))

# 棋子描边色变体池（v39）：App 域解剖实锤绿描边棋子系统性漏检（f17 三格绿圈白底，
# cnvcs 画风）——渲染分布从未覆盖"绿色描边"。描边色 12% 概率走变体池。
RIM_ALT = ((40, 130, 80), (60, 140, 90), (90, 90, 95), (150, 60, 50))

# v18 布局靶向偏置开关（蒸馏集生成时：通用集关、靶向集开；训练在线渲染默认开）
V18_BIAS = os.environ.get("RENDERER_V18_BIAS", "1") == "1"

SS = 2  # 超采样倍率：小字号文字渲染质量的关键，所有 flat 坐标 ×SS

# ==================== 布局：随机对局模拟（伪合法走子，仅视觉用途） ====================
def start_layout() -> dict:
    """标准开局（canonical：红在 row 9，row 0 为黑方底线）。

    v18 修复存量 bug：back 为小写，row 9 原样写入导致红方底线十个字全是
    黑子小写——v2~v17 所有合成数据里红 K/A/B/R/N 从未存在（红方仅剩
    2 炮 5 兵），模型对红背排的认知只来自真实帧，"底排红子认成黑子"
    （bench (9,7) N/R 互翻、App 域底排整行分歧）由此而来。"""
    lay = {}
    back = "rnbakabnr"
    for c, ch in enumerate(back):
        lay[(0, c)] = ch.lower()
        lay[(9, c)] = ch.upper()
    for c in (1, 7):
        lay[(2, c)] = "c"; lay[(7, c)] = "C"
    for c in (0, 2, 4, 6, 8):
        lay[(3, c)] = "p"; lay[(6, c)] = "P"
    return lay

def in_palace(r: int, c: int, red: bool) -> bool:
    return (3 <= c <= 5) and (7 <= r <= 9 if red else 0 <= r <= 2)

def own_half(r: int, red: bool) -> bool:
    return r >= 5 if red else r <= 4

def piece_moves(lay: dict, pos: tuple) -> list:
    """伪合法走位（不管将军/照面——布局只供视觉训练，不要求对局合法性）"""
    r, c = pos
    ch = lay[pos]
    red = ch.isupper()
    out = []
    if ch in "Rr":  # 车：直线滑动
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS:
                if (nr, nc) in lay:
                    if lay[(nr, nc)].isupper() != red:
                        out.append((nr, nc))
                    break
                out.append((nr, nc)); nr += dr; nc += dc
    elif ch in "Nn":  # 马：八向跳+蹩马腿
        for dr, dc in ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)):
            nr, nc = r + dr, c + dc
            leg = (r + (1 if dr == 2 else -1 if dr == -2 else 0),
                   c + (1 if dc == 2 else -1 if dc == -2 else 0))
            if 0 <= nr < ROWS and 0 <= nc < COLS and leg not in lay:
                if (nr, nc) not in lay or lay[(nr, nc)].isupper() != red:
                    out.append((nr, nc))
    elif ch in "Cc":  # 炮：平移同车、吃子须隔一子
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            jumped = False
            while 0 <= nr < ROWS and 0 <= nc < COLS:
                if (nr, nc) in lay:
                    if not jumped:
                        jumped = True
                    else:
                        if lay[(nr, nc)].isupper() != red:
                            out.append((nr, nc))
                        break
                elif not jumped:
                    out.append((nr, nc))
                nr += dr; nc += dc
    elif ch in "Kk":  # 帅/将：九宫直行一步
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if in_palace(nr, nc, red) and ((nr, nc) not in lay or lay[(nr, nc)].isupper() != red):
                out.append((nr, nc))
    elif ch in "Aa":  # 仕/士：九宫斜行一步
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            if in_palace(nr, nc, red) and ((nr, nc) not in lay or lay[(nr, nc)].isupper() != red):
                out.append((nr, nc))
    elif ch in "Bb":  # 相/象：田字不过河、塞象眼
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            nr, nc = r + dr, c + dc
            eye = (r + dr // 2, c + dc // 2)
            if 0 <= nr < ROWS and 0 <= nc < COLS and own_half(nr, red) and eye not in lay:
                if (nr, nc) not in lay or lay[(nr, nc)].isupper() != red:
                    out.append((nr, nc))
    elif ch in "Pp":  # 兵/卒：过河前只进、过河后可横移
        dr = -1 if red else 1
        cands = [(r + dr, c)]
        if (r <= 4) if red else (r >= 5):
            cands += [(r, c - 1), (r, c + 1)]
        for nr, nc in cands:
            if 0 <= nr < ROWS and 0 <= nc < COLS and ((nr, nc) not in lay or lay[(nr, nc)].isupper() != red):
                out.append((nr, nc))
    return out

def random_layout(rng: random.Random, stage: str) -> dict:
    """随机布局：opening(0~6 步)/midgame(15~60 步)/endgame(40~80 步+随机摘子保双王)"""
    lay = start_layout()
    moves = {"opening": rng.randint(0, 6), "midgame": rng.randint(15, 60),
             "endgame": rng.randint(40, 80)}[stage]
    red_go = True
    for _ in range(moves):
        side = str.isupper if red_go else str.islower
        froms = [p for p, ch in lay.items() if side(ch) and piece_moves(lay, p)]
        if not froms:
            break
        src = rng.choice(froms)
        dsts = piece_moves(lay, src)
        if not dsts:
            continue
        dst = rng.choice(dsts)
        ch = lay.pop(src)
        lay[dst] = ch  # 目标格有敌子即吃子覆盖
        red_go = not red_go
    if stage == "endgame":
        pieces = [p for p, ch in lay.items() if ch.upper() != "K"]
        rng.shuffle(pieces)
        for p in pieces[: rng.randint(10, 18)]:
            n_red = sum(1 for ch in lay.values() if ch.isupper())
            n_blk = sum(1 for ch in lay.values() if ch.islower())
            if n_red > 1 and n_blk > 1:
                del lay[p]
    # 定向补样（v10b）：bench 残余错格的布局长尾——沉底黑炮 (9,0)×6 漏检。
    # 走子模拟中炮沉底角落密度 <1%，8% 概率强制补（覆盖不影响标签自洽）。
    if stage != "opening" and rng.random() < 0.08:
        pos = (9, rng.choice([0, 1, 7, 8]))
        ch_here = lay.get(pos)
        if ch_here is None or ch_here.isupper():
            lay[pos] = "c"
    # 定向补样（v18）：v17 bench 残余 18 错格收敛出的 6 类系统性混淆——
    # 黑 r/n 背排 (0,1)/(0,7) 互认、(9,7) N/R 互翻、(9,8) 空格幻视 R、
    # (9,6)B→p、(8,5)/(9,5) p→c/A。靶向样本由 teacher（渲染域 100%）打标签，
    # 零噪声且量无上限；王位 (0,4)/(9,4) 不在靶点集，覆盖不破坏双王自洽。
    # RENDERER_V18_BIAS=0 关闭（通用蒸馏集用），默认开（靶向集/训练在线渲染用）。
    if V18_BIAS and stage != "opening" and rng.random() < 0.30:
        r = rng.random()
        if r < 0.25:
            # 黑车马背排互认对：(0,1)=n 与 (0,7)=r 强制共存（开局典型位）
            for pos, ch in (((0, 1), "n"), ((0, 7), "r")):
                if lay.get(pos) != "k":
                    lay[pos] = ch
        elif r < 0.55:
            # v41 底排右角靶向（第三次修正）：侦察否决 20% 翻转（真实对局 (9,7) 车仅
            # 占 9.1%——20% 会过矫正），改为强投 (9,7)=R 至渲染占比 ~9% 匹配真实锚点；
            # (9,8) 保持自然（车位格先验本就均衡 43% 车），(9,7)=N 系反例保留
            sub = rng.random()
            if sub < 0.65:
                if lay.get((9, 7)) != "K":
                    lay[(9, 7)] = "R"
                if rng.random() < 0.5:
                    lay.pop((9, 8), None)   # 一半配空格（.→R 幻视反例）、一半保持自然
            elif sub < 0.85:
                if lay.get((9, 7)) != "K":
                    lay[(9, 7)] = "N"
                lay.pop((9, 8), None)
            else:
                if lay.get((9, 8)) != "K":
                    lay[(9, 8)] = "R"
                lay.pop((9, 7), None)
        elif r < 0.60:
            if lay.get((9, 6)) != "K":
                lay[(9, 6)] = "B"
        elif r < 0.80:
            pos = (8, 5) if rng.random() < 0.5 else (9, 5)
            if lay.get(pos) != "K":
                lay[pos] = "p"
        elif r < 0.88:
            # v39 红马定向（App 域解剖：N→R ×5 最大真错簇，App 描边画风红马/傌）：
            # 红马强投至中下场任意位（含底排），每子独立选 馬/傌 繁体变体
            pos = (rng.choice([6, 7, 8, 9]), rng.randint(0, 8))
            if lay.get(pos) != "K":
                lay[pos] = "N"
        # 其余 20% 不强制，保持自然分布
    return lay

# ==================== 主题与棋子外观 ====================
@dataclass
class Theme:
    bg: tuple            # 棋盘底色
    line: tuple          # 网格线颜色
    river_text: tuple    # 楚河汉界文字颜色
    red_face: tuple      # 红子盘面
    black_face: tuple    # 黑子盘面
    red_text: tuple      # 红子文字
    black_text: tuple    # 黑子文字
    name: str = ""

THEMES = [
    Theme((222, 184, 140), (92, 60, 35), (120, 85, 55), (246, 232, 205), (246, 232, 205),
          (198, 32, 32), (38, 34, 32), "classic_wood"),
    Theme((232, 200, 154), (107, 69, 37), (150, 105, 60), (242, 228, 200), (242, 228, 200),
          (168, 32, 32), (32, 32, 32), "cnvcs_wood"),
    Theme((112, 76, 50), (226, 214, 192), (210, 196, 172), (66, 55, 48), (66, 55, 48),
          (255, 84, 70), (238, 236, 232), "dark_wood"),
    Theme((236, 229, 216), (72, 72, 74), (150, 145, 138), (250, 248, 242), (250, 248, 242),
          (205, 35, 35), (42, 42, 44), "light_plain"),
    Theme((46, 49, 57), (152, 157, 168), (120, 126, 138), (72, 77, 88), (72, 77, 88),
          (255, 96, 82), (242, 242, 242), "dark_plain"),
    Theme((62, 112, 82), (232, 232, 222), (200, 210, 200), (246, 240, 226), (246, 240, 226),
          (200, 34, 34), (40, 40, 40), "green_felt"),
    # 教学图/习题图画风（26 帧人工真值 App 域主画像）：米黄底+细网格+描边圆圈细字
    Theme((233, 207, 161), (94, 62, 32), (146, 108, 66), (249, 243, 229), (249, 243, 229),
          (188, 38, 38), (28, 28, 28), "worksheet_wood"),
    Theme((240, 233, 218), (110, 100, 86), (160, 150, 134), (252, 249, 241), (252, 249, 241),
          (178, 34, 34), (34, 34, 34), "worksheet_plain"),
    # v50 蓝白教学主题（26 帧 f20/f21/f22 主画像：米白底+浅蓝细线+白/浅蓝圆片蓝字，
    # 红黑双方同用蓝色字——App 教学图不区分红黑字色；直打 v2 口径字形混淆 7 格的画风域）
    Theme((246, 249, 253), (148, 168, 198), (155, 175, 205), (252, 253, 255), (225, 237, 250),
          (28, 84, 178), (28, 84, 178), "blue_white_worksheet"),
]
# 主题采样权重：木质亮色系是真实 App 的绝对主流（cnvcs/象棋迪克均如此），加权采样；
# worksheet 两主题对应教学图/残局习题图（App 域错格 67% 集中在底排红子的画风域）
THEME_WEIGHTS = [2, 4, 1, 1, 1, 1, 2, 1, 3]

def jitter_color(rng: random.Random, col: tuple, amt: int = 14) -> tuple:
    return tuple(max(0, min(255, v + rng.randint(-amt, amt))) for v in col)

@dataclass
class StylePack:
    """一次采样的全部外观随机量（渲染可复现的关键：全部来自 rng）"""
    theme: Theme
    fonts: list = field(default_factory=list)
    line_w: float = 2.0
    piece_r: float = 19.5
    disc_rim: bool = True
    disc_inner: bool = True
    river_text_on: bool = True
    grid_jitter: float = 0.0
    disc_3d: bool = False      # 立体盘面（高光偏移+暗边）
    markers_on: bool = False   # 炮/兵位十字位标
    text_scale: float = 1.4    # 字号相对棋子半径

_FONT_CACHE: dict = {}

def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(path, size)
        _FONT_CACHE[key] = f
    return f

def _random_theme(rng: random.Random) -> Theme:
    """v33 随机板式:HSV 采样生成全新配色(30% 样本替换固定主题池,泛化路线)。
    可读性约束:底色中低饱和中亮、线色与底色拉开明度、子盘面亮于底色;
    红黑子墨色保持现实分布(红系/深系),不随机。名字带 wood 后缀时渲染木纹。"""
    import colorsys

    def rgb(hh: float, ss: float, vv: float) -> tuple:
        return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hh % 1.0, ss, min(1.0, vv)))

    h, s, v = rng.random(), rng.uniform(0.06, 0.42), rng.uniform(0.42, 0.94)
    # v55 暗盘分支：15% 概率压到深色域（街头夜摊/深色木盘——钉子户考古 #12 纯黑格
    # 幻视的渲染域空白：随机板式此前最暗 v=0.42，覆盖不到近黑底色）
    if rng.random() < 0.15:
        v = rng.uniform(0.14, 0.32)
    bg = rgb(h, s, v)
    line = rgb(h + rng.uniform(-0.06, 0.06), min(1.0, s * rng.uniform(0.5, 1.3)),
               v * (rng.uniform(0.30, 0.55) if rng.random() < 0.5 else rng.uniform(1.25, 1.45)))
    face = rgb(h, s * 0.55, min(1.0, v * rng.uniform(1.18, 1.42)))
    name = "gen_random_wood" if rng.random() < 0.4 else "gen_random"
    return Theme(bg, line, jitter_color(rng, line, 20), face, face,
                 (198, 32, 32), (38, 34, 32), name)


# v38 提速：MaxFilter/MinFilter → cv2.dilate/erode（3×3 核 + BORDER_REPLICATE，
# 与 PIL 逐位一致——.cache 等价性实测 True；cv2 内核更快，省 ~11ms/样本）
_MORPH_K3 = np.ones((3, 3), np.uint8)


def _max3(img: Image.Image) -> Image.Image:
    return Image.fromarray(cv2.dilate(np.array(img), _MORPH_K3, borderType=cv2.BORDER_REPLICATE))


def _min3(img: Image.Image) -> Image.Image:
    return Image.fromarray(cv2.erode(np.array(img), _MORPH_K3, borderType=cv2.BORDER_REPLICATE))


def sample_style(rng: random.Random, font_files: list) -> StylePack:
    if rng.random() < 0.30:                       # v33:30% 概率随机板式(泛化)
        st = StylePack(theme=_random_theme(rng))
    else:
        st = StylePack(theme=rng.choices(THEMES, weights=THEME_WEIGHTS)[0])
    st.fonts = rng.sample(font_files, k=min(3, len(font_files)))
    st.line_w = rng.uniform(1.6, 2.8)
    st.piece_r = rng.uniform(17.8, 21.2)   # warp 空间棋子半径（格子间距 ~44）
    st.disc_rim = rng.random() < 0.85
    st.disc_inner = rng.random() < 0.6
    st.river_text_on = rng.random() < 0.9
    st.grid_jitter = rng.uniform(0.0, 0.6)
    st.disc_3d = rng.random() < 0.5
    st.markers_on = rng.random() < 0.6
    st.text_scale = rng.uniform(1.3, 1.62)
    return st

# ==================== 棋盘平面渲染（warp 空间 ×SS） ====================
def _wood_grain(img: Image.Image, rng: random.Random, base: tuple) -> None:
    """程序化木纹：随机半透明横向条纹"""
    w, h = img.size
    grain = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(grain)
    darker = tuple(max(0, v - rng.randint(8, 22)) for v in base)
    for _ in range(rng.randint(14, 30)):
        y = rng.randint(0, h)
        d.line([(0, y), (w, y + rng.randint(-6, 6))], fill=darker + (rng.randint(10, 36),),
               width=rng.randint(1, 3) * SS)
    img.alpha_composite(grain)

def render_board_flat(lay: dict, st: StylePack, rng: random.Random) -> Image.Image:
    """渲染 warp 空间棋盘（×SS 超采样）RGBA。网格角恒在 (50,50)/(400,50)/(400,450)/(50,450)（×SS）。"""
    W, H = WARP_W * SS, WARP_H * SS
    bg = jitter_color(rng, st.theme.bg, 16)
    img = Image.new("RGBA", (W, H), bg + (255,))
    if "wood" in st.theme.name:
        _wood_grain(img, rng, bg)
    d = ImageDraw.Draw(img)
    lw = max(1, round(st.line_w * SS))
    line = jitter_color(rng, st.theme.line, 12)

    def gx(c): return (col_x(c) + rng.uniform(-st.grid_jitter, st.grid_jitter)) * SS
    def gy(r): return (row_y(r) + rng.uniform(-st.grid_jitter, st.grid_jitter)) * SS

    # 横线 10 条（贯通）
    for r in range(ROWS):
        d.line([(gx(0), gy(r)), (gx(COLS - 1), gy(r))], fill=line, width=lw)
    # 竖线 9 条：中间 7 条在楚河汉界断开（row4~row5），两侧贯通
    for c in range(COLS):
        if c in (0, COLS - 1):
            d.line([(gx(c), gy(0)), (gx(c), gy(ROWS - 1))], fill=line, width=lw)
        else:
            d.line([(gx(c), gy(0)), (gx(c), gy(4))], fill=line, width=lw)
            d.line([(gx(c), gy(5)), (gx(c), gy(ROWS - 1))], fill=line, width=lw)
    # 九宫斜线
    for (c0, r0, c1, r1) in ((3, 0, 5, 2), (5, 0, 3, 2), (3, 7, 5, 9), (5, 7, 3, 9)):
        d.line([(gx(c0), gy(r0)), (gx(c1), gy(r1))], fill=line, width=lw)
    # 外框（常见样式：网格外再加一圈粗框）
    pad2 = 6 * SS
    d.rectangle([gx(0) - pad2, gy(0) - pad2, gx(COLS - 1) + pad2, gy(ROWS - 1) + pad2],
                outline=line, width=max(1, lw + SS))

    # 楚河 汉界
    if st.river_text_on and st.fonts:
        try:
            f = _load_font(st.fonts[0], 21 * SS)
            rt = st.theme.river_text + (200,)
            mid_y = (gy(4) + gy(5)) / 2
            d.text(((gx(1) + gx(3)) / 2, mid_y), "楚  河", font=f, fill=rt, anchor="mm")
            d.text(((gx(5) + gx(7)) / 2, mid_y), "漢  界", font=f, fill=rt, anchor="mm")
        except Exception:
            pass

    # 炮/兵位十字位标（真实棋盘常见样式）
    if st.markers_on:
        tick = 3.2 * SS
        for (mc, mr) in ((1, 2), (7, 2), (1, 7), (7, 7), (0, 3), (2, 3), (4, 3), (6, 3), (8, 3),
                         (0, 6), (2, 6), (4, 6), (6, 6), (8, 6)):
            x0, y0 = gx(mc), gy(mr)
            for sxx in (-1, 1):
                for syy in (-1, 1):
                    d.line([(x0 + sxx * tick * 1.6, y0 + syy * tick * 0.6),
                            (x0 + sxx * tick * 0.6, y0 + syy * tick * 0.6),
                            (x0 + sxx * tick * 0.6, y0 + syy * tick * 1.6)],
                           fill=line, width=max(1, SS))

    # v55 格心十字定位线盘式（练习盘/教学盘常见样式：每格中心印小十字——钉子户
    # 考古 #14/#15/#16 幻视格均为盘面十字/竖线纹理区，渲染此前零覆盖。印在盘面，
    # 有子格被棋子自然盖住，与实物一致）
    if rng.random() < 0.14:
        tick = rng.uniform(3.0, 5.0) * SS
        lw2 = max(1, SS)
        for r in range(ROWS):
            for c in range(COLS):
                x0, y0 = gx(c), gy(r)
                d.line([(x0 - tick, y0), (x0 + tick, y0)], fill=line, width=lw2)
                d.line([(x0, y0 - tick), (x0, y0 + tick)], fill=line, width=lw2)

    # 网格交点粗块（v10b）：cnvcs 等真实 App 的线端加粗样式（bench 拼图实测交点有粗黑块）
    if rng.random() < 0.5:
        for r in range(ROWS):
            for c in range(COLS):
                if rng.random() < 0.35:
                    s = rng.uniform(1.2, 3.2) * SS
                    d.rectangle([gx(c) - s, gy(r) - s, gx(c) + s, gy(r) + s], fill=line)

    # 棋子
    # 实物域增广(v30→v33 双通道):
    # ①各向同性抖动(25%,v30 原有):人手摆子偏离交叉点至半格、相邻子可部分重叠(路边摊
    #   实拍帧00/05 漏判病灶);标签仍记交叉点(人工标注=就近格)。
    # ②视差场(30%,v33 新增):立体棋子顶面在残差倾角下相对格心发生"方向一致"的径向偏移
    #   ——消失点(相机正下方在盘面的投影,可盘外)随机,整盘偏移方向相干、幅度∝tan(倾角)
    #   ×子高系数×离心度。各向同性抖动覆盖不了这种方向结构,"透视压扁"病灶即此;泛化
    #   基线实测 nano 对 σ16px 残差只掉 1pt 而我方掉 7.3pt,带宽与方向结构双缺口。
    # 截图域(cnvcs/worksheet)真实帧子恒居中,两通道概率门控不影响该域(真实帧不入渲染器)。
    piece_off = rng.random() < 0.25
    off_x_cap = (WARP_W - 2 * WARP_PAD) / (COLS - 1) * 0.4 * SS   # 0.4 格距(warp 空间)
    off_y_cap = (WARP_H - 2 * WARP_PAD) / (ROWS - 1) * 0.4 * SS
    parallax = rng.random() < 0.30
    if parallax:
        theta = min(abs(rng.gauss(0, math.radians(7))), math.radians(18))
        px_nx = rng.uniform(-0.5, 1.5) * WARP_W          # 消失点 x(warp 像素,可盘外)
        px_ny = rng.uniform(-0.5, 1.5) * WARP_H
        px_gain = math.tan(theta) * rng.uniform(0.30, 0.45) * 1.2
    light_xy = None                                      # 单光源方向(逐样本惰性采样)
    # v55 棋子投影：65% 样本整盘开影子（方向随单光源，钉子户考古漏检 2 格均为暗背
    # 景+棋子无影"融入盘面"；渲染此前只有棋子级高光/暗晕、盘面上没有落影）
    shadow_on = rng.random() < 0.65
    shadow_alpha = rng.randint(45, 110)
    shadow_blur = rng.uniform(1.2, 3.2)
    shadow_dist = rng.uniform(0.20, 0.50)
    for (r, c), ch in lay.items():
        cx, cy = col_x(c) * SS, row_y(r) * SS
        if piece_off:
            cx += rng.uniform(-off_x_cap, off_x_cap)
            cy += rng.uniform(-off_y_cap, off_y_cap)
        if parallax:
            vx, vy = cx - px_nx * SS, cy - px_ny * SS
            dist = math.hypot(vx, vy) + 1e-6
            mag = px_gain * min(dist / (0.9 * WARP_W * SS), 1.5) * 44 * SS
            cx += vx / dist * mag
            cy += vy / dist * mag
        R_ = st.piece_r * SS
        if shadow_on:
            # 落影在光源反侧，先于棋子绘制（压在盘面上、被棋子盖住中心）
            if light_xy is None:
                light_xy = (rng.uniform(-1, 1), rng.uniform(-1, 1))
            sr = R_ * rng.uniform(0.85, 1.10)
            sx = cx - light_xy[0] * R_ * shadow_dist
            sy = cy - light_xy[1] * R_ * shadow_dist
            pad_s = sr * 1.8
            sh = Image.new("RGBA", (max(2, int(2 * pad_s)), max(2, int(2 * pad_s))), (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse([pad_s - sr, pad_s - sr, pad_s + sr, pad_s + sr],
                                       fill=(20, 16, 12, shadow_alpha))
            sh = sh.filter(ImageFilter.GaussianBlur(radius=shadow_blur * SS))
            img.alpha_composite(sh, (int(sx - pad_s), int(sy - pad_s)))
        red = ch.isupper()
        face = jitter_color(rng, st.theme.red_face if red else st.theme.black_face, 10)
        rim = jitter_color(rng, st.theme.line, 20)
        if rng.random() < 0.12:                       # v39 描边色变体（含绿，App 域解剖修复）
            rim = jitter_color(rng, rng.choice(RIM_ALT), 14)
        if not red and rng.random() < 0.12:
            # v57 黑子暗面档：暗光下黑棋子盘面压暗 35~55%（钉子户 (15,2,8) b 六轮
            # 漏检的对症现象——黑子在暗背景上对比度过低；仅压 face 不动描边/字色）
            face = tuple(max(0, int(v * rng.uniform(0.45, 0.65))) for v in face)
        d.ellipse([cx - R_, cy - R_, cx + R_, cy + R_], fill=face + (255,),
                  outline=rim + (255,), width=max(1, SS))
        if rng.random() < 0.55:
            # v33 棋子级光照:单光源高光斑+对侧暗晕(实物棋子摄照观感;渲染平面盘面此前
            # 只有描边弧,与实拍的柔和高光/阴影差距是 piece 轴 occ 0.44 的成分之一)
            if light_xy is None:
                light_xy = (rng.uniform(-1, 1), rng.uniform(-1, 1))
            for sgn, alpha, col in ((1, rng.randint(30, 62), (255, 255, 255)),
                                    (-1, rng.randint(22, 46), (0, 0, 0))):
                hr = R_ * rng.uniform(0.5, 0.72)
                hx = cx + light_xy[0] * R_ * 0.5 * sgn
                hy = cy + light_xy[1] * R_ * 0.5 * sgn
                ov = Image.new("RGBA", (max(2, int(2 * hr)), max(2, int(2 * hr))), (0, 0, 0, 0))
                ImageDraw.Draw(ov).ellipse([0, 0, 2 * hr - 1, 2 * hr - 1], fill=col + (alpha,))
                img.alpha_composite(ov, (int(hx - hr), int(hy - hr)))
        if st.disc_3d:
            # 立体感：左上高光弧 + 右下暗边（两道偏移椭圆弧近似）
            hl = tuple(min(255, v + rng.randint(22, 40)) for v in face)
            dk = tuple(max(0, v - rng.randint(18, 34)) for v in face)
            d.arc([cx - R_ + SS, cy - R_ + SS, cx + R_ - SS, cy + R_ - SS],
                  start=140, end=320, fill=hl + (255,), width=2 * SS)
            d.arc([cx - R_ + SS, cy - R_ + SS, cx + R_ - SS, cy + R_ - SS],
                  start=-40, end=140, fill=dk + (255,), width=2 * SS)
        if st.disc_inner:
            r2 = R_ * rng.uniform(0.72, 0.82)
            d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2],
                      outline=jitter_color(rng, st.theme.line, 26) + (255,), width=SS)
        if st.fonts:
            try:
                fpath = rng.choice(st.fonts)
                fsize = int(R_ * st.text_scale * rng.uniform(0.94, 1.06))
                f = _load_font(fpath, fsize)
                if red:
                    base = st.theme.red_text if rng.random() < 0.72 else rng.choice(RED_INK_ALT)
                else:
                    base = st.theme.black_text if rng.random() < 0.82 else rng.choice(BLACK_INK_ALT)
                tcol = jitter_color(rng, base, 22)
                pad = fsize
                tmp = Image.new("RGBA", (pad * 2, pad * 2), (0, 0, 0, 0))
                td = ImageDraw.Draw(tmp)
                td.text((pad, pad), glyph_of(ch, rng, fpath), font=f, fill=tcol + (255,), anchor="mm")
                if rng.random() < 0.8:
                    tmp = tmp.rotate(rng.uniform(-8, 8), resample=Image.BICUBIC)
                # v33 字形增广:笔画粗细/弹性波/斜切(实物棋子刻字与印刷体的形态差距;
                # piece 轴基线我方 occ 0.44 vs nano 0.81,字形形态鲁棒性是缺口之一)
                g = rng.random()
                if g < 0.18:
                    tmp = _max3(tmp)                                     # 加粗(阴刻/粗铸)
                elif g < 0.32:
                    tmp = _min3(tmp)                                     # 变细(细刻/磨损)
                elif g < 0.45:
                    arr = np.array(tmp)                              # 弹性波(手工刻字不直)
                    h2, w2 = arr.shape[:2]
                    amp = rng.uniform(0.8, 1.8) * SS
                    fy, fx = np.meshgrid(np.arange(h2), np.arange(w2), indexing="ij")
                    wav = rng.choice([1, 2])
                    dx = (amp * np.sin(2 * math.pi * wav * fy / h2 + rng.uniform(0, 6))).astype(np.float32)
                    dy = (amp * np.sin(2 * math.pi * wav * fx / w2 + rng.uniform(0, 6))).astype(np.float32)
                    out = cv2.remap(arr, (fx + dx).astype(np.float32), (fy + dy).astype(np.float32),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    tmp = Image.fromarray(out)
                elif g < 0.58:
                    sh = rng.uniform(-0.18, 0.18)                    # 斜切(拍摄透视的局部残余)
                    tmp = tmp.transform(tmp.size, Image.AFFINE, (1, sh, -sh * pad, 0, 1, 0),
                                        resample=Image.BICUBIC)
                img.alpha_composite(tmp, (int(cx - pad), int(cy - pad)))
            except Exception:
                pass

    # cnvcs 最后走子标记增广（v10b）：红色虚线方框——终点标记压在刚走子的棋子上、
    # 起点标记留在空位（bench 每帧都有；bench 错格 (9,8)空→車幻检/黑炮漏检与此相关）。
    # 标记不改变棋子身份 → labels 不动，学生学"红虚线框=噪声忽略"。
    if rng.random() < 0.40:
        occupied = list(lay.keys())
        empty = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in lay]
        pools = ([occupied, empty] if rng.random() < 0.6 else [empty, occupied])
        for pool in pools:
            if not pool:
                continue
            r, c = rng.choice(pool)
            cx, cy = col_x(c) * SS, row_y(r) * SS
            half = st.piece_r * SS * rng.uniform(0.92, 1.12)
            col = jitter_color(rng, (208, 36, 28), 18) + (255,)
            seg = max(1, int(half / 3))
            for (x0, y0, x1, y1) in ((cx - half, cy - half, cx + half, cy - half),
                                     (cx + half, cy - half, cx + half, cy + half),
                                     (cx + half, cy + half, cx - half, cy + half),
                                     (cx - half, cy + half, cx - half, cy - half)):
                for s in range(4):  # 四边各 3 段虚线（画 1 跳 1）
                    if s % 2:
                        continue
                    t0, t1 = s / 4, (s + 1) / 4
                    d.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                            (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
                           fill=col, width=max(1, 2 * SS))
    return img

# ==================== 整屏合成（背景+透视+遮挡；只渲染棋盘邻域局部画布） ====================
def synth_screen(flat: Image.Image, rng: random.Random, screen_w: int, screen_h: int):
    """把平面棋盘经随机单应贴到虚拟屏幕的棋盘邻域。
    返回 (region BGR, 网格四角[4,2]局部坐标 TL,TR,BR,BL, 遮挡矩形列表(局部), Hm(flat2x网格角→局部))。
    screen_w/h 只作为「虚拟屏幕」尺度上下文（决定棋盘占屏比例与遮挡物尺寸），
    实际渲染仅覆盖棋盘 bbox+边距的局部区域——分类样本只需要棋盘周围，整屏渲染是浪费。"""
    # 背景渐变（虚拟屏幕纵向渐变，局部画布按绝对行号取值）
    c0 = (rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245))
    c1 = (rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245))

    # 网格四边形（虚拟屏幕系）：高度占屏 55%~92%，旋转 ±14°，梯形透视扰动
    frac = rng.uniform(0.55, 0.92)
    bh = screen_h * frac
    bw = min(bh * (WARP_W - 2 * WARP_PAD) / (WARP_H - 2 * WARP_PAD), screen_w * 0.96)
    theta = math.radians(rng.uniform(-14, 14))
    ct, st_ = math.cos(theta), math.sin(theta)
    cx0 = rng.uniform(bw / 2 + 8, screen_w - bw / 2 - 8)
    cy0 = rng.uniform(bh / 2 + 8, screen_h - bh / 2 - 8)
    hw, hh = bw / 2, bh / 2
    kst = rng.uniform(0.0, 0.05)
    quad_rel = np.array([
        [-hw, -hh],
        [hw, -hh * (1 - kst * rng.uniform(0, 1))],
        [hw, hh],
        [-hw, hh * (1 - kst * rng.uniform(0, 1))],
    ], np.float32)
    rot = np.array([[ct, -st_], [st_, ct]], np.float32)
    quad = quad_rel @ rot.T + np.array([cx0, cy0], np.float32)
    quad = quad + np.float32(rng.uniform(-3, 3))  # 整体平移微扰

    # Hm：flat(2x) 网格角 → 虚拟屏幕四角（TL,TR,BR,BL）
    src = np.array([[50, 50], [400, 50], [400, 450], [50, 450]], np.float32) * SS
    Hm_full = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

    def apply_h(px: float, py: float) -> list:
        v = Hm_full @ np.array([px, py, 1.0], np.float64)
        return [v[0] / v[2], v[1] / v[2]]

    # flat 画布四角（含 50px 木框边距）在屏幕上的落点 → 画布级单应
    dst_canvas = np.array([apply_h(0, 0), apply_h(WARP_W * SS, 0),
                           apply_h(WARP_W * SS, WARP_H * SS), apply_h(0, WARP_H * SS)], np.float32)
    H_canvas = cv2.getPerspectiveTransform(
        np.array([[0, 0], [WARP_W * SS, 0], [WARP_W * SS, WARP_H * SS], [0, WARP_H * SS]], np.float32),
        dst_canvas)

    # 局部画布 = 棋盘网格 bbox + 边距（覆盖木框出画与遮挡需要）
    margin = int(bh * 0.18) + 40
    x0 = int(max(0, quad[:, 0].min() - margin))
    y0 = int(max(0, quad[:, 1].min() - margin))
    x1 = int(min(screen_w, quad[:, 0].max() + margin))
    y1 = int(min(screen_h, quad[:, 1].max() + margin))
    W_loc, H_loc = x1 - x0, y1 - y0
    off = np.array([x0, y0], np.float32)

    # 背景渐变（局部，按绝对行号取值，保证与虚拟屏幕语义一致）
    rows_abs = (np.arange(y0, y1, dtype=np.float32) / max(1, screen_h - 1)).reshape(-1, 1, 1)
    bg = (np.array(c0, np.float32) * (1 - rows_abs) + np.array(c1, np.float32) * rows_abs)
    bg = np.repeat(bg, W_loc, axis=1)

    # warp 到局部画布（dst 平移 offset）
    flat_bgra = cv2.cvtColor(np.array(flat), cv2.COLOR_RGBA2BGRA)
    H_local = H_canvas.copy()
    H_local[0:2, 2] -= off  # 齐次矩阵平移列减 offset
    warped = cv2.warpPerspective(flat_bgra, H_local, (W_loc, H_loc),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    out = (bg * (1 - alpha) + warped[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    # 遮挡物（对话框/悬浮面板样式）0~2 个，局部坐标系，可能压住棋盘边缘
    occluders = []
    for _ in range(rng.randint(0, 2)):
        if rng.random() < 0.45:
            ow = rng.randint(screen_w // 6, screen_w // 2)
            oh = rng.randint(screen_h // 12, screen_h // 5)
            ox = rng.randint(-ow // 3, max(0, W_loc - ow // 2))
            oy = rng.randint(0, max(0, H_loc - oh))
            col = rng.choice([(240, 240, 240), (30, 30, 34), (250, 245, 235), (60, 62, 70)])
            cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), col, -1)
            cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), (120, 120, 120), 2)
            occluders.append((ox, oy, ox + ow, oy + oh))
    return out, (quad - off).astype(np.float32), occluders, _shift_h(Hm_full, off)

def _shift_h(H: np.ndarray, off: np.ndarray) -> np.ndarray:
    """把 flat→屏幕 的单应改为 flat→局部（屏幕点整体平移 -off）"""
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = -off[0]; T[1, 2] = -off[1]
    return (T @ H).astype(np.float32)

# ==================== 背景字增广（v50：打实物域背景字幻视） ====================
# 26 帧 v2 口径幻视 6 格中 4 格是背景文字（f11"海底捞"×2、f16 大字水印×2）——渲染器
# 此前无任何文字背景（grep 实证）。水印随盘面视为背景，标签不动。
WM_TEXTS = ["海底捞", "休闲娱乐", "茶楼棋牌", "中国象棋", "残局挑战", "朋友的店",
            "CHESS CLUB", "TOURNAMENT"]
_WM_FONTS: list | None = None   # 延迟初始化的字体路径缓存（load_font_pool 定义在后部）


def watermark_text(w450: np.ndarray, rng: random.Random, font_paths: list) -> np.ndarray:
    """p≈0.25 叠 1-2 条低 alpha（0.25~0.6）中英水印横穿盘面。颜色用 BGR 序
    （w450 由 RGBA2BGRA warp 而来）；白/黑通道对称，红字直接给 BGR 值。"""
    global _WM_FONTS
    if _WM_FONTS is None:
        _WM_FONTS = load_font_pool()          # 延迟初始化（load_font_pool 定义在后部）
    font_paths = _WM_FONTS
    if rng.random() >= 0.25 or not font_paths:
        return w450
    h, w = w450.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(rng.randint(1, 2)):
        try:
            # v55 字号双档 + v57 巨字入镜档：大字招牌（h//9~h//5）+ 小字桌布/垫布
            # （h//16~h//10）+ 巨字招牌入镜（h//4~h//3，钉子户考古 f16"街"字入镜
            # 幻视两格的对症现象——大幅面高饱和红字从画面边缘压进盘面）
            u = rng.random()
            if u < 0.30:
                fs = rng.randint(max(10, h // 16), max(12, h // 10))
            elif u < 0.85:
                fs = rng.randint(max(14, h // 9), max(16, h // 5))
            else:
                fs = rng.randint(max(20, h // 4), max(24, h // 3))
            f = _load_font(rng.choice(font_paths), fs)
        except OSError:
            return w450
        t = rng.choice(WM_TEXTS)
        if fs > h // 4:                       # 巨字档用 1-2 字（招牌竖排字入镜形态）
            t = rng.choice(["街", "棋", "牌", "茶", "局", "帥", "中國象棋"])
        tw = int(draw.textlength(t, font=f))
        x = rng.randint(-tw // 3, max(1, w - tw // 2))
        y = rng.randint(-h // 10, max(1, h - f.size))
        a = rng.randint(64, 180)   # v55 上限 150→180（更实的招牌字）
        col = rng.choice([(255, 255, 255), (45, 45, 45), (50, 60, 190), (60, 60, 235)])  # BGR：白/黑/暗红/亮红
        if fs > h // 4:
            col = rng.choice([(60, 60, 235), (50, 60, 190), (45, 45, 45)])  # 巨字档偏红/黑实色
            a = rng.randint(130, 200)
        draw.text((x, y), t, font=f, fill=col + (a,))
    # v55 旋转档：±28° 斜排字（实物招牌/贴纸 rarely 水平）
    if rng.random() < 0.5:
        layer = layer.rotate(rng.uniform(-28, 28), resample=Image.BICUBIC, expand=False)
    mask = np.asarray(layer.split()[3], np.float32) / 255.0
    rgb_layer = np.asarray(layer.convert("RGB"), np.float32)
    m3 = mask[..., None]
    out3 = (rgb_layer * m3 + w450[..., :3].astype(np.float32) * (1 - m3)).astype(np.uint8)
    return np.dstack([out3, np.full((h, w, 1), 255, np.uint8)])   # 保持 BGRA 4 通道


# ==================== 光度扰动（LUT 实现，作用在 warp 回采后的小图上） ====================
def board_illumination(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """v55 棋盘级光照不均：径向暗角 + 单侧渐变（红队实证：photometric 此前纯全局
    LUT 无空间项——实物照片的边缘压暗/单侧阴影入镜是渲染域零覆盖维度，幻视与
    漏检的共因）。乘法增益场，不改标签。"""
    h, w = img.shape[:2]
    if rng.random() < 0.35:      # 径向暗角：光心随机偏移，边缘压到 floor
        vx, vy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
        sigma = rng.uniform(0.55, 0.95) * max(w, h)
        floor = rng.uniform(0.35, 0.65)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = 1.0 - (1.0 - floor) * np.clip(np.hypot(xx - vx, yy - vy) / sigma, 0.0, 1.0) ** 2
        img = np.clip(img.astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)
    if rng.random() < 0.30:      # 单侧渐变：窗光/阴影入镜
        axis, sign = rng.choice([(0, 1), (0, -1), (1, 1), (1, -1)])
        lo = rng.uniform(0.45, 0.75)
        n = h if axis == 0 else w
        t = np.linspace(lo, 1.0, n).astype(np.float32)
        if sign < 0:
            t = t[::-1]
        g = t[:, None] if axis == 0 else t[None, :]
        img = np.clip(img.astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)
    return img


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """亮度/对比度/gamma/色温合成 256 级查找表一次施加（比全图浮点 gamma 快一个量级），
    再叠加可选模糊与高斯噪声。"""
    gain = rng.uniform(0.82, 1.18)
    bias = rng.uniform(-14, 14)
    gamma = rng.uniform(0.8, 1.25)
    xs = np.arange(256, dtype=np.float32)
    base = np.clip(xs * gain + bias, 0, 255)
    base = np.clip(255.0 * np.power(base / 255.0, gamma), 0, 255)
    chans = [cv2.LUT(img[:, :, i],
                     np.clip(base * rng.uniform(0.92, 1.08), 0, 255).astype(np.uint8))
             for i in range(3)]
    out = cv2.merge(chans)
    if rng.random() < 0.5:
        k = rng.choice([1, 1, 3])
        if k > 1:
            out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.3, 0.9))
    if rng.random() < 0.6:  # 高斯噪声
        noise = np.zeros_like(out, np.float32)
        cv2.randn(noise, 0, rng.uniform(2, 6))
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def jpeg_roundtrip(img: np.ndarray, rng: random.Random) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 96)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img

# ==================== 样本产出（分类输入 + 标注） ====================
GRID_SRC = np.array([[50, 50], [400, 50], [400, 450], [50, 450]], np.float32)

def warp_back_to_warp450(screen_bgr: np.ndarray, quad: np.ndarray, rng: random.Random,
                         corner_jitter: bool = True) -> np.ndarray:
    """整屏帧 → warp 空间 450×500 BGR（复刻端上单应采样；corner_jitter 模拟 RTMPose 定位误差）"""
    q = quad.astype(np.float32).copy()
    if corner_jitter:
        sigma = rng.choices([2.0, 4.0, 8.0], weights=[0.6, 0.3, 0.1])[0]
        q = q + np.array([[rng.gauss(0, sigma), rng.gauss(0, sigma)] for _ in range(4)], np.float32)
    H = cv2.getPerspectiveTransform(q, GRID_SRC)
    return cv2.warpPerspective(screen_bgr, H, (WARP_W, WARP_H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def warp450_to_cls(warp450_bgr: np.ndarray) -> np.ndarray:
    """450×500 → 中心裁 400×450 → 缩放 280×315 RGB（端上 classifyCells 的裁剪缩放链）"""
    crop = warp450_bgr[25:475, 25:425]
    rs = cv2.resize(crop, (CLS_IN_W, CLS_IN_H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)

def labels_of(lay: dict, occluders: list) -> np.ndarray:
    """90 格类别标注。occluders 为 warp450 坐标系矩形（快速路径在 warp 后绘制），
    格子中心 (col_x,row_y) 落入矩形即视为被遮挡 → IGNORE_LABEL。"""
    labels = np.full(ROWS * COLS, CLASS_INDEX["."], np.int64)
    occ = np.zeros(ROWS * COLS, bool)
    for r in range(ROWS):
        for c in range(COLS):
            ch = lay.get((r, c))
            if ch:
                labels[r * COLS + c] = CLASS_INDEX[ch]
            if occluders:
                px, py = col_x(c), row_y(r)
                for (x0, y0, x1, y1) in occluders:
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        occ[r * COLS + c] = True
    labels[occ] = IGNORE_LABEL
    return labels

def make_sample(rng: random.Random, font_files: list, stage: str | None = None,
                screen_size: tuple = (1080, 2280)):
    """产出一条分类样本（快速路径）：
    cls_img (uint8 RGB 280×315) / labels (90,) / w450 (warp 空间 BGR) / layout / stage / theme。

    快速路径说明：分类器的输入分布 = 端上 warp 之后的「canonical 棋盘」——现实中的
    倾斜已被人应变换去除，几何扰动只需模拟姿态估计误差（角点抖动+微小旋转/梯形残差），
    因此直接 flat→warp450 单次 warp（0.225Mpx），无需合成整屏再采回（省 ~90ms/样本）。
    遮挡物在 warp 空间直接绘制（屏内矩形经逆变换本就是四边形，物类等价）。
    整屏合成路径（synth_screen）保留给二期角点模型的整屏帧训练数据。"""
    stage = stage or rng.choices(["opening", "midgame", "endgame"], weights=[1, 2, 1])[0]
    lay = random_layout(rng, stage)
    style = sample_style(rng, font_files)
    flat = render_board_flat(lay, style, rng)
    w450 = warp_flat_to_warp450(flat, rng)
    # 遮挡物（warp450 坐标）：模拟对话框/悬浮窗压住棋盘边缘。
    # v39 加灰面板变体（App 域解剖 f23 灰色竖条遮挡）与半透明样式（PIL 级 alpha 混合太贵，
    # 用"浅灰实心+白边"近似半透明观感）
    occluders = []
    for _ in range(rng.randint(0, 2)):
        if rng.random() < 0.45:
            ow = rng.randint(70, 230)
            oh = rng.randint(70, 230)
            ox = rng.randint(-ow // 2, WARP_W - ow // 2)
            oy = rng.randint(-oh // 2, WARP_H - oh // 2)
            if rng.random() < 0.3:
                col = (rng.randint(150, 185),) * 3      # 灰面板（近似半透明悬浮层）
                cv2.rectangle(w450, (ox, oy), (ox + ow, oy + oh), col, -1)
                cv2.rectangle(w450, (ox, oy), (ox + ow, oy + oh), (230, 230, 230), 2)
            else:
                col = rng.choice([(240, 240, 240), (30, 30, 34), (250, 245, 235), (60, 62, 70)])
                cv2.rectangle(w450, (ox, oy), (ox + ow, oy + oh), col, -1)
                cv2.rectangle(w450, (ox, oy), (ox + ow, oy + oh), (120, 120, 120), 2)
            occluders.append((ox, oy, ox + ow, oy + oh))
    # v57 盘边阴影入画档：盘缘/垫布/邻物在贴片边缘形成 12~32px 暗带（钉子户
    # (05,3,8) C 六轮漏检的对症现象——棋子骑格心+贴片边缘黑边切边）。带矩形
    # 追加进 occluders：覆盖到格心的格 IGNORE，避免"盖住棋子仍标占子"的错标。
    if rng.random() < 0.08:
        side = rng.randrange(4)
        bw = rng.randint(12, 32)
        am = rng.uniform(0.35, 0.60)
        dark = np.array([8, 6, 5, 255], np.float32)   # BGR+A 深黑
        sl = (slice(0, bw), slice(None)) if side == 0 else \
             (slice(WARP_H - bw, WARP_H), slice(None)) if side == 1 else \
             (slice(None), slice(0, bw)) if side == 2 else \
             (slice(None), slice(WARP_W - bw, WARP_W))
        w450[sl] = (w450[sl].astype(np.float32) * (1 - am) + dark * am).astype(np.uint8)
        if side == 0:
            occluders.append((0, 0, WARP_W, bw))
        elif side == 1:
            occluders.append((0, WARP_H - bw, WARP_W, WARP_H))
        elif side == 2:
            occluders.append((0, 0, bw, WARP_H))
        else:
            occluders.append((WARP_W - bw, 0, WARP_W, WARP_H))
    w450 = watermark_text(w450, rng, _WM_FONTS)
    w450 = board_illumination(w450, rng)   # v55 棋盘级光照不均（暗角/单侧渐变）
    if rng.random() < 0.5:
        w450 = jpeg_roundtrip(photometric(w450, rng), rng)
    else:
        w450 = photometric(w450, rng)
    cls_img = warp450_to_cls(w450)
    labels = labels_of(lay, occluders)
    return {"cls_img": cls_img, "labels": labels, "w450": w450,
            "layout": lay, "stage": stage, "theme": style.theme.name}

# 姿态尺度增广范围（双轴独立，网格中心不动）。v2 默认 0.94~1.06；v3 下探 0.85
# 覆盖 RTMPose 外框语义错位带（一期实测最优校正 0.91~0.94，真实标注链含 0.92 域）
SCALE_RANGE = (0.94, 1.06)


def set_scale_range(lo: float, hi: float):
    """训练侧调整尺度增广下探（train.py --scale-lo 调用）"""
    global SCALE_RANGE
    SCALE_RANGE = (lo, hi)


def warp_flat_to_warp450(flat: Image.Image, rng: random.Random,
                         corner_jitter: bool = True,
                         sigma_corner: float | None = None,
                         sigma_rot: float | None = None,
                         sigma_trap: float | None = None) -> np.ndarray:
    """flat(2x RGBA) → warp450 BGR，单次 warp。
    目标四角 = GRID_SRC + 姿态误差模拟：高斯角点抖动 σ∈{2,4,8,12,16}px（模拟 RTMPose 定位
    误差，v33 带宽校准）+ 绕中心微小旋转 N(0,2.5°) + 梯形残差 N(0,2.5%)（倾斜画面经姿态
    纠平后的残余）。sigma_corner/sigma_rot/sigma_trap 可强制指定严重度（v33 泛化评测阶梯
    与带宽校准用），缺省 None = 训练分布内部抽样，行为与历史版本一致。
    corner_jitter=False 时输出几何精确的 canonical 视图（标注链验证用）。"""
    src = np.array([[50, 50], [400, 50], [400, 450], [50, 450]], np.float32) * SS
    dst = GRID_SRC.copy().astype(np.float64)
    # 系统性姿态偏差增广（关键！）：实测 RTMPose 在真实 App 上存在 ~(+18,+14)px 的
    # 一致性偏移（约 40% 格距，见 docs 评测报告）——现役 nano_v3 训练分布含此偏差故免疫。
    # 这里用均匀全局偏移 ±24px + 中心尺度 ±3% 覆盖之，叠加逐角抖动/微旋转/梯形残差。
    dst += np.array([rng.uniform(-24, 24), rng.uniform(-24, 24)])  # 全局偏移（双轴独立）
    center0 = np.array([(50 + 400) / 2, (50 + 450) / 2])
    # bench 型失真模式（30% 密度）：复刻 bench 实测 pose 框-网格关系——框比网格宽
    # 且中心偏移 → 内容在 warp 域缩小并偏移。v7（固定 1.029/1.005+右下偏）已验证该
    # 方向有效（bench occ 83→88%）；v8 扩为随机幅度。v9 再定向：平移偏向"内容出界"
    # （左列左移 / 右列右移 / 顶行上移，双方向交替），专练边缘格的出界容差——
    # v8 残余 28 错格全部集中在边缘格位（0/8 列、9 行）。
    if rng.random() < 0.30:
        ex = rng.uniform(1.02, 1.05)
        ey = rng.uniform(1.00, 1.02)
        dx = rng.choice([-1.0, 1.0]) * rng.uniform(2, 16)
        dy = rng.choice([-1.0, 1.0]) * rng.uniform(2, 12)
        dst = (dst - center0) * np.array([ex, ey]) + center0 + np.array([dx, dy])
    # 非对称双轴尺度：v3 起下探可配（SCALE_RANGE），覆盖外框语义错位带
    sx = rng.uniform(*SCALE_RANGE)
    sy = rng.uniform(*SCALE_RANGE)
    dst = (dst - center0) * np.array([sx, sy]) + center0
    if corner_jitter:
        # v33 带宽校准:{2,4,8}→{2,4,8,12,16}(重加权),旋转 σ1.2°→2.5°,梯形 σ1%→2.5%
        # 依据:泛化基线实测 nano 在 σ16px+5° 下只掉 1pt,我方掉 7.3pt——真实 RTMPose
        # 残差带宽(实物域"透视压扁"病灶)超出旧模拟带,{12,16} 档即为此补。
        # v36 加码:高档位采样加权 [0.35,0.30,0.20,0.10,0.05]→[0.22,0.26,0.24,0.16,0.12]
        # (R3+R4 占比 15%→28%)——angle 轴 R4 0.5269 是 v35 最弱档;带宽本身不动
        # (带宽抬升会引入标签错位噪声,只挪采样重心不抬上限)。
        sigma = sigma_corner if sigma_corner is not None else rng.choices(
            [2.0, 4.0, 8.0, 12.0, 16.0], weights=[0.22, 0.26, 0.24, 0.16, 0.12])[0]
        dst += np.array([[rng.gauss(0, sigma), rng.gauss(0, sigma)] for _ in range(4)], np.float64)
        ang = math.radians(rng.gauss(0, sigma_rot if sigma_rot is not None else 2.5))
        ct, st_ = math.cos(ang), math.sin(ang)
        center = np.array([WARP_W / 2, WARP_H / 2])
        dst = (dst - center) @ np.array([[ct, -st_], [st_, ct]], np.float64) + center
        kest = rng.gauss(0, (sigma_trap if sigma_trap is not None else 0.025)) * (WARP_H - 2 * WARP_PAD)
        dst[1, 1] += kest * rng.uniform(0, 1)   # TR 角
        dst[3, 1] -= kest * rng.uniform(0, 1)   # BL 角
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    flat_bgra = cv2.cvtColor(np.array(flat), cv2.COLOR_RGBA2BGRA)
    return cv2.warpPerspective(flat_bgra, H, (WARP_W, WARP_H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# ==================== 自检：接触表（人工抽检入口） ====================
def load_font_pool(recog_dir: Path | None = None) -> list:
    d = recog_dir or Path(__file__).parent
    return sorted(str(p) for p in list(d.glob("fonts/*.ttf")) + list(d.glob("fonts/*.otf")))

def contact_sheet(font_files: list, n: int = 24, out: str | None = None,
                  seed: int = 2026) -> str:
    out = out or str(Path(__file__).parent / "data" / "contact_sheet.png")
    rng = random.Random(seed)
    cols_n = 6
    rows_n = (n + cols_n - 1) // cols_n
    cell_w, cell_h = CLS_IN_W + 8, CLS_IN_H + 8
    sheet = Image.new("RGB", (cols_n * cell_w, rows_n * cell_h), (24, 24, 28))
    n_occluded = 0
    for i in range(n):
        s = make_sample(rng, font_files)
        im = Image.fromarray(s["cls_img"])
        sheet.paste(im, ((i % cols_n) * cell_w + 4, (i // cols_n) * cell_h + 4))
        if int((s["labels"] == IGNORE_LABEL).sum()) > 0:
            n_occluded += 1
            ImageDraw.Draw(sheet).rectangle(
                [(i % cols_n) * cell_w + 2, (i // cols_n) * cell_h + 2,
                 (i % cols_n) * cell_w + 10, (i // cols_n) * cell_h + 10], fill=(255, 80, 40))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"接触表: {out}（{n} 样本，含遮挡样本 {n_occluded} 个，左上红点标记遮挡格）")
    return out

if __name__ == "__main__":
    fonts = load_font_pool()
    print(f"字体池: {len(fonts)} 个")
    contact_sheet(fonts, n=24)
