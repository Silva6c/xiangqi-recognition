# -*- coding: utf-8 -*-
"""cell_cls 数据管线（v63）：三源混合单格样本供给。

源构成（每批随机混合）：
  A. 在线渲染高清格（~60%）：render_board_flat（×SS=2，格 87×89px 原生高清）→ 抠格
     → 格级增广。一次渲染产 90 格，摊薄后 ~0.5ms/样本。
  B. 真实占子贴片（~28%）：vn/zfk/efbok 三库 ~50.8K 枚（38×38，真实字形多样性），
     按类均衡抽取。
  C. 真实空格贴片（~12%）：real_empty_patch_lib_v50 24084 片（真实空格纹理，
     幻视对症——cell_cls 必须把真实空格判得极准）。

格级增广包（128×128 域）：随机仿射（旋转/平移/尺度，模拟 warp 残差与棋子骑格心）、
光度抖动、低清模拟（降采样 32~56px 再升回，覆盖 bench 域上采样分布）、边缘暗带
（切边/盘边阴影对症）、遮挡块、半透明背景字（幻视对症）、JPEG。

红线：GEN_HOLDOUT 3 字体剔除不进训练（与 train.py 同口径）；26 帧评测集/eval_* npz
永不入训（本管线不读任何 eval 文件）。
"""
import random
import threading
import queue

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

import renderer as R
from cell_model import CELL_IN, NCLASS

CLS_CHARS = R.CLASS_CHARS  # ".xKABNRCPkabnrcp"
GEN_HOLDOUT = {"NotoSerifSC-Regular.ttf", "ZCOOLXiaoWei-Regular.ttf", "SourceHanSerifCN-Bold.otf"}


# ---------------- 真实贴片库 ----------------

def load_patch_libs(recog_dir="."):
    """载入三个占子贴片库 + 空格库。返回 {cls_idx: [N,38,38,3]} 合并字典 + 空格数组。"""
    occ = {}
    for name in ("vn_raw_patch_lib", "zfk_raw_patch_lib", "efbok_patch_lib"):
        d = np.load(f"{recog_dir}/data/{name}.npz", allow_pickle=True)
        for k in d.files:
            if k.startswith("cls"):
                occ.setdefault(int(k[3:]), []).append(d[k])
    occ = {c: np.concatenate(v) for c, v in occ.items()}
    empty = np.load(f"{recog_dir}/data/real_empty_patch_lib_v50.npz")["patches"]
    return occ, empty


# ---------------- 格级增广 ----------------

def _affine(img: Image.Image, rng: random.Random) -> Image.Image:
    """旋转 ±10°、平移 ±12%、尺度 0.85~1.18（棋子骑格心/偏移/warp 残差模拟）。"""
    ang = rng.uniform(-10, 10)
    scale = rng.uniform(0.85, 1.18)
    dx, dy = rng.uniform(-0.12, 0.12) * CELL_IN, rng.uniform(-0.12, 0.12) * CELL_IN
    cos, sin = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    m = [cos * scale, -sin * scale, dx + (1 - cos * scale) * CELL_IN / 2 + sin * scale * CELL_IN / 2,
         sin * scale, cos * scale, dy + (1 - cos * scale) * CELL_IN / 2 - sin * scale * CELL_IN / 2]
    return img.transform((CELL_IN, CELL_IN), Image.AFFINE, m, resample=Image.BILINEAR,
                         fillcolor=(rng.randint(28, 72),) * 3)


def _photometric(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.72, 1.32))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.35))
    return img


def _dark_band(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """一侧 12~32px 暗带（v57 盘边阴影/切边格对症——f05(3,8) 六轮漏检现象）。"""
    side = rng.randrange(4)
    bw = rng.randint(12, 32)
    am = rng.uniform(0.35, 0.60)
    sl = [slice(None)] * 2
    sl[0 if side < 2 else 1] = slice(0, bw) if side % 2 == 0 else slice(CELL_IN - bw, CELL_IN)
    arr[tuple(sl)] = (arr[tuple(sl)].astype(np.float32) * (1 - am)).astype(np.uint8)
    return arr


def _occlude(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """随机矩形遮挡（边缘区域、≤30% 边长——不全盖棋子避免标签噪声）。"""
    w = rng.randint(20, 38)
    x = rng.choice([0, CELL_IN - w]) if rng.random() < 0.7 else rng.randint(0, CELL_IN - w)
    y = rng.randint(0, CELL_IN - w) if x in (0, CELL_IN - w) and rng.random() < 0.7 else rng.choice([0, CELL_IN - w])
    col = tuple(rng.randint(24, 216) for _ in range(3))
    cv2.rectangle(arr, (x, y), (x + w, y + w), col, -1)
    return arr


_WM_CACHE = {}


def _bg_text(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """半透明背景大字叠加（f11/f16 背景字幻视对症：背景有字仍判空格/棋子）。
    字画在角部、alpha 低——不遮棋子主体，标签保持有效。"""
    if not _WM_CACHE:
        for f in ("C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"):
            try:
                _WM_CACHE[f] = ImageFont.truetype(f, 64)
                break
            except OSError:
                continue
    font = next(iter(_WM_CACHE.values()), None)
    if font is None:
        return arr
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img, "RGBA")
    ch = rng.choice("海底捞街棋象棋残局中盘胜")
    corner = rng.choice([(0, 0), (CELL_IN - 66, 0), (0, CELL_IN - 70), (CELL_IN - 66, CELL_IN - 70)])
    col = rng.choice([(200, 30, 30), (30, 30, 200), (20, 20, 20)])
    dr.text(corner, ch, font=font, fill=(*col, rng.randint(70, 130)))
    return np.array(img)


def augment(pil: Image.Image, rng: random.Random) -> np.ndarray:
    """完整增广链：仿射 → 光度 → (暗带/遮挡/背景字) → 低清模拟 → JPEG。"""
    img = _affine(pil, rng)
    img = _photometric(img, rng)
    arr = np.array(img)
    if rng.random() < 0.15:
        arr = _dark_band(arr, rng)
    if rng.random() < 0.10:
        arr = _occlude(arr, rng)
    if rng.random() < 0.08:
        arr = _bg_text(arr, rng)
    if rng.random() < 0.30:  # 低清模拟：bench/底片域上采样分布
        small = rng.randint(32, 56)
        arr = cv2.resize(cv2.resize(arr, (small, small), interpolation=cv2.INTER_AREA),
                         (CELL_IN, CELL_IN), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.20:
        arr = np.array(Image.fromarray(arr).resize((74, 74)).resize((CELL_IN, CELL_IN), Image.LANCZOS))
        enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 88)])[1]
        arr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return arr


# ---------------- 源 A：渲染高清格 ----------------

def render_board_cells(rng: random.Random, fonts: list):
    """渲染一整盘（SS=2 高清 flat）并抠出 90 格 [(PIL Image, label)]。
    不走 make_sample：flat 上无遮挡/光度（那些由格级增广包覆盖），换来 ×4 面积的
    原生高清（87px vs w450 的 44px）。"""
    lay = R.random_layout(rng, rng.choices(["opening", "midgame", "endgame"], weights=[1, 2, 1])[0])
    style = R.sample_style(rng, fonts)
    flat = R.render_board_flat(lay, style, rng)          # RGBA (WARP_W*SS, WARP_H*SS)
    cells = []
    for (r, c), ch in lay.items():
        cx, cy = R.col_x(c) * R.SS, R.row_y(r) * R.SS    # flat 空间格心（≈87×89px/格）
        half = 44 * R.SS  # 抠 88px 方形（格 87×89，稍大于一格）
        box = (cx - half + rng.uniform(-6, 6), cy - half + rng.uniform(-6, 6),
               cx + half + rng.uniform(-6, 6), cy + half + rng.uniform(-6, 6))
        crop = flat.crop(tuple(int(v) for v in box)).convert("RGB").resize((CELL_IN, CELL_IN), Image.LANCZOS)
        cells.append((crop, R.CLASS_INDEX[ch]))
    # 空格格：从同一盘面补采（布局外的格=空格，取网格内随机位）
    empty_pts = [(r, c) for r in range(R.ROWS) for c in range(R.COLS) if (r, c) not in lay]
    for r, c in rng.sample(empty_pts, min(len(empty_pts), len(lay))):
        cx, cy = R.col_x(c) * R.SS, R.row_y(r) * R.SS
        half = 44 * R.SS
        crop = flat.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))
                         ).convert("RGB").resize((CELL_IN, CELL_IN), Image.LANCZOS)
        cells.append((crop, 0))
    return cells


# ---------------- 真实格 bank（v2：真实域精度主弹药） ----------------

def load_banks(recog_dir="."):
    """载入 LO/HD 两个真实格 bank（不存在则跳过返回 None）。"""
    banks = {}
    for tag, fn in (("lo", "cell_bank_lo.npz"), ("hd", "cell_bank_hd.npz")):
        p = f"{recog_dir}/data/{fn}"
        try:
            d = np.load(p, allow_pickle=True)
            banks[tag] = (d["cells"], d["labels"])
        except FileNotFoundError:
            banks[tag] = None
    return banks


# ---------------- 混合批供给器 ----------------

class CellFeeder:
    """线程化预取：渲染盘（重）与贴片/bank 抽样（轻）混合出批。

    源配比（bank 缺失时自动归一化到现存源）：
      渲染高清 30% / LO bank 30% / HD bank 15% / 占子贴片 17% / 空格贴片 8%
    """

    def __init__(self, seed: int, batch_size: int, workers: int = 2):
        self.rng = random.Random(seed)
        self.bs = batch_size
        self.fonts = [f for f in R.load_font_pool() if f.split("\\")[-1].split("/")[-1] not in GEN_HOLDOUT]
        self.occ, self.empty = load_patch_libs()
        self.occ_cls = sorted(self.occ.keys())
        self.banks = load_banks()
        srcs = {"render": 30, "lo": 30, "hd": 15, "occ": 17, "empty": 8}
        srcs = {k: v for k, v in srcs.items() if k == "render" or k == "occ" or k == "empty"
                or self.banks.get(k) is not None}
        tot = sum(srcs.values())
        self.weights = list(srcs.values())
        self.keys = list(srcs.keys())
        self.weights = [w / tot for w in self.weights]
        self.lock = threading.Lock()
        self.q = queue.Queue(maxsize=6)
        self.stop = False
        self.threads = [threading.Thread(target=self._fill, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def _one_sample(self, wr: random.Random):
        src = wr.choices(self.keys, weights=self.weights)[0]
        if src == "render":
            if not hasattr(wr, "_cells") or not wr._cells:
                wr._cells = render_board_cells(wr, self.fonts)
            return wr._cells.pop(wr.randrange(len(wr._cells)))
        if src in ("lo", "hd"):
            cells, labels = self.banks[src]
            for _ in range(8):  # x 伪标签类跳过；HD 空格 50% 降权；帅/将稀有类直取（v2 误伤 f14 K→. 对症）
                i = wr.randrange(len(cells))
                lb = labels[i]
                if lb == 1:
                    continue
                if lb in (2, 9):
                    break
                if src == "hd" and lb == 0 and wr.random() < 0.5:
                    continue
                break
            img = Image.fromarray(cells[i]).resize((CELL_IN, CELL_IN), Image.LANCZOS)
            return img, int(labels[i])
        if src == "occ":
            ci = wr.choice(self.occ_cls)
            bank = self.occ[ci]
            img = Image.fromarray(bank[wr.randrange(len(bank))]).resize((CELL_IN, CELL_IN), Image.LANCZOS)
            return img, ci
        img = Image.fromarray(self.empty[wr.randrange(len(self.empty))]
                              ).resize((CELL_IN, CELL_IN), Image.LANCZOS)
        return img, 0

    def _fill(self):
        wr = random.Random(self.rng.randrange(1 << 30))
        while not self.stop:
            imgs, labs = [], []
            for _ in range(self.bs):
                pil, lab = self._one_sample(wr)
                imgs.append(augment(pil, wr))
                labs.append(lab)
            batch = (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255),
                     torch.tensor(labs, dtype=torch.long))
            self.q.put(batch, timeout=120)

    def next(self):
        return self.q.get(timeout=120)

    def close(self):
        self.stop = True
