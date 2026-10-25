# -*- coding: utf-8 -*-
"""角点模型训练数据工厂（v64）。

三源混合（CellFeeder 线程化骨架：2 worker 预取 + queue）：
  A. 渲染整屏源（主力）：corner_place（gen_screen30.place_board_fullframe 几何族
     加宽版：±18°/占比 0.50~0.90/透视 0~0.06，种子独立于考卷 20261025 体系），
     GT=渲染自证（解析角点，零标注）。一盘 flat 多次贴放（几何/背景/遮挡逐帧重采）
     摊薄渲染成本。
  B. 真实源（第 1 波注入）：RTMPose 伪标签帧（.cache/corner_pseudo.npz，
     排除清单见 pseudo_label 脚本），按需读图 + 同构 affine。
  C. 负样本（~8%）：合成"无棋盘"整屏（渐变/单色背景 + 界面色块 + 非棋盘比例细网格），
     target=均匀分布——教模型压平峰值，由端上质量门/boxPlausible 拦截。

输入构造与端上逐项同构：pose_affine(w,h)（帧宽×1.25 中心方形，pose_corner.py 与
OnnxBoardRecognizer.poseAffine 逐项对齐的实现）→ 256×256 越界补黑；角点 GT 同仿射
进 256 空间。本模块角点序一律 [TL,TR,BL,BR]（模型/端上序；place 返回 TL,TR,BR,BL
需转序）。

红线：GEN_HOLDOUT 3 字体剔除；eval_* npz / screen30_v1（seed 20261025）永不入训
（渲染源用独立随机种子体系，与考卷种子空间不相交）。
"""
import math
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

import renderer as R
from cell_data import GEN_HOLDOUT
from cell_model import CELL_IN  # noqa: F401  （保持与 cell 系同源习惯）
from corner_model import CORNER_IN, NKP
from pose_corner import IMAGENET_MEAN, IMAGENET_STD, corner_affine

SCREEN = (1080, 2280)  # 与 screen30 考卷同屏（端上竖屏分布；照片域宽高比由真实源覆盖）


def imread_cn(path):
    """Windows 中文路径安全读图（cv2.imread 在非 ASCII 路径下静默返回 None——
    photo_crawl 池约 7 成帧是中文文件名，实测 3746 张全被误杀）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------- 整屏贴放（gen_screen30 几何族加宽版） ----------------

def corner_place(flat, rng, rot_max=35.0, bh_lo=0.30, bh_hi=0.90, kst_max=0.48):
    """flat（900×1000 RGBA，含 50px 木框）→ 1080×2280 整屏 + 网格四角 GT。

    与 gen_screen30.place_board_fullframe 同几何族但参数域加宽（考卷封版 ±14°/0.55~0.85/
    0.05；训练域 ≥ 考卷域）；返回 quad [TL,TR,BR,BL]（调用方负责转模型序）。
    v3 修订：kst 0.06→0.40、rot 18→25——v2 heldout 目检实锤街拍 30~60° 强透视斜拍
    是 photo 域 gap 主因（同盘系统性偏移），渲染分布必须覆盖斜拍形态。
    v4 修订：bh_lo 0.50→0.30——真机录屏实锤小棋盘场景（复盘 App 棋盘仅占屏 37%，
    v3 全段定位失败），渲染分布补"小棋盘+大背景"形态。
    v5 修订：kst 0.55、rot 35——nano17 有效局目检实锤街拍 40~50° 俯角+大旋转超出
    v4 域（定位 1~2cm 偏→warp 切格歪→识别崩），继续加宽斜拍形态。
    v6 修订：kst 0.55→0.48 折中——v5 加宽挤压效应（7/8 号 ok→FAIL、screen30 环3
    92→96），回撤一档求"保 11/12/16 救回+回收 7/8+缓仿真退步"两全。
    """
    grid_w, grid_h = 350.0, 400.0
    bh = SCREEN[1] * rng.uniform(bh_lo, bh_hi)
    bw = bh * grid_w / grid_h
    theta = math.radians(rng.uniform(-rot_max, rot_max))
    ct, st_ = math.cos(theta), math.sin(theta)
    kst = rng.uniform(0.0, kst_max)
    hw, hh = bw / 2, bh / 2
    quad_rel = np.array([[-hw, -hh], [hw, -hh * (1 - kst)], [hw, hh],
                         [-hw, hh * (1 - kst)]], np.float32)
    rot = np.array([[ct, -st_], [st_, ct]], np.float32)
    quad_rot = quad_rel @ rot.T
    aabb_w = quad_rot[:, 0].max() - quad_rot[:, 0].min()
    aabb_h = quad_rot[:, 1].max() - quad_rot[:, 1].min()
    s = min(1.0, (SCREEN[0] * 0.92) / aabb_w, (SCREEN[1] * 0.92) / aabb_h)
    quad_rel *= s
    hw, hh = hw * s, hh * s
    quad_rot = quad_rel @ rot.T
    aabb_w = quad_rot[:, 0].max() - quad_rot[:, 0].min()
    aabb_h = quad_rot[:, 1].max() - quad_rot[:, 1].min()
    cx0 = rng.uniform(aabb_w / 2 + 10, SCREEN[0] - aabb_w / 2 - 10)
    cy0 = rng.uniform(aabb_h / 2 + 10, SCREEN[1] - aabb_h / 2 - 10)
    quad = quad_rot + np.array([cx0, cy0], np.float32)   # [TL,TR,BR,BL]

    SS = R.SS
    src_grid = np.array([[50, 50], [400, 50], [400, 450], [50, 450]], np.float32) * SS
    Hm = cv2.getPerspectiveTransform(src_grid, quad.astype(np.float32))

    def apply_h(px_, py_):
        v = Hm @ np.array([px_, py_, 1.0], np.float64)
        return [v[0] / v[2], v[1] / v[2]]

    dst_canvas = np.array([apply_h(0, 0), apply_h(900.0, 0),
                           apply_h(900.0, 1000.0), apply_h(0, 1000.0)], np.float32)
    # 局部 warp（棋盘 AABB clip 屏内）+ 贴背景，代替全屏 warpPerspective（渲染提速主项）
    x0 = max(int(np.floor(dst_canvas[:, 0].min())) - 4, 0)
    x1 = min(int(np.ceil(dst_canvas[:, 0].max())) + 4, SCREEN[0])
    y0 = max(int(np.floor(dst_canvas[:, 1].min())) - 4, 0)
    y1 = min(int(np.ceil(dst_canvas[:, 1].max())) + 4, SCREEN[1])
    H_canvas = cv2.getPerspectiveTransform(
        np.array([[0, 0], [900.0, 0], [900.0, 1000.0], [0, 1000.0]], np.float32),
        (dst_canvas - np.array([x0, y0], np.float32)).astype(np.float32))

    # 背景（uint8 直算，免 float32 全屏 28MB 分配——渲染提速主项之二）：
    # 60% 双色渐变 / 25% 单色 / 15% 单色+噪点（考卷只有渐变，训练域加宽）
    # v5 修订：单色分支 60% 概率叠加"软件 UI 界面"几何块（标题栏/侧面板/按钮排/
    # 头像圆——nano17 实锤桌面软件/手机整机截图形态塌缩 FAIL，渲染分布补 UI 背景）
    mode = rng.random()
    noisy = mode >= 0.85
    if mode < 0.60:
        c0a = np.array((rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245)), np.int32)
        c1a = np.array((rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245)), np.int32)
        rows = np.arange(SCREEN[1], dtype=np.int32) * 255 // (SCREEN[1] - 1)      # [H] 0..255
        bg1 = ((c0a[None] * (255 - rows[:, None]) + c1a[None] * rows[:, None]) // 255
               ).astype(np.uint8)                                                # [H,3]
        out = np.repeat(bg1[:, None, :], SCREEN[0], axis=1)
    else:
        c0a = np.array((rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245)), np.int32)
        out = np.tile(c0a.astype(np.uint8), (SCREEN[1], SCREEN[0], 1))
        if rng.random() < 0.60:                   # 软件 UI 界面背景
            def ui_color():
                return (rng.randint(30, 235), rng.randint(30, 235), rng.randint(30, 235))
            if rng.random() < 0.6:                # 标题栏
                bar_h = rng.randint(40, 140)
                cv2.rectangle(out, (0, 0), (SCREEN[0], bar_h), ui_color(), -1)
            for _ in range(rng.randint(2, 4)):    # 侧面板块
                pw = rng.randint(80, 260)
                x_side = rng.choice((0, SCREEN[0] - pw))
                cv2.rectangle(out, (x_side, rng.randint(0, 200)),
                              (x_side + pw, SCREEN[1] - rng.randint(0, 300)), ui_color(), -1)
            for _ in range(rng.randint(3, 7)):    # 按钮排（底部一排小矩形）
                bw_, bh_ = rng.randint(60, 160), rng.randint(36, 80)
                bx = rng.randint(0, max(1, SCREEN[0] - bw_))
                by = rng.randint(SCREEN[1] // 2, max(SCREEN[1] // 2 + 1, SCREEN[1] - bh_))
                cv2.rectangle(out, (bx, by), (min(bx + bw_, SCREEN[0]), min(by + bh_, SCREEN[1])),
                              ui_color(), -1)
            for _ in range(rng.randint(1, 3)):    # 头像圆
                cv2.circle(out, (rng.randint(0, SCREEN[0]), rng.randint(0, SCREEN[1] // 3)),
                           rng.randint(30, 70), ui_color(), -1)
    flat_bgra = cv2.cvtColor(np.array(flat), cv2.COLOR_RGBA2BGRA)
    warped = cv2.warpPerspective(flat_bgra, H_canvas, (x1 - x0, y1 - y0),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    a = warped[:, :, 3:4].astype(np.int16)
    roi = out[y0:y1, x0:x1].astype(np.int32)
    w3 = warped[:, :, :3].astype(np.int32)
    roi = (roi * (255 - a) + w3 * a + 127) // 255
    if noisy:  # 噪点只加 ROI（全屏噪声 float 开销大且多在棋盘外无意义）
        roi = np.clip(roi + np.random.RandomState(rng.randrange(1 << 30)).normal(
            0, 5, roi.shape[:2] + (1,)).astype(np.int32), 0, 255)
    out[y0:y1, x0:x1] = roi.astype(np.uint8)

    # 遮挡（对话框样式，样式对齐考卷）
    for _ in range(rng.randint(0, 2)):
        if rng.random() < 0.45:
            ow = rng.randint(SCREEN[0] // 6, SCREEN[0] // 2)
            oh = rng.randint(SCREEN[1] // 12, SCREEN[1] // 5)
            ox = rng.randint(-ow // 3, max(0, SCREEN[0] - ow // 2))
            oy = rng.randint(0, max(0, SCREEN[1] - oh))
            col = rng.choice([(240, 240, 240), (30, 30, 34), (250, 245, 235), (60, 62, 70)])
            cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), col, -1)
            cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), (120, 120, 120), 2)
    return out, quad


def make_negative_frame(rng):
    """合成"无棋盘"整屏：渐变/单色背景 + 0~5 界面色块 + 20% 非棋盘比例细网格。

    目的：教模型在无棋盘帧上压平 SimCC 峰值（→ 均匀分布 target），端上由
    质量门/boxPlausible 拦截——"宁可 null 不可错点"语义。非棋盘网格线是
    近似棋盘格线的专门迷惑项。
    """
    mode = rng.random()
    if mode < 0.5:
        c0 = (rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245))
        c1 = (rng.randint(20, 245), rng.randint(20, 245), rng.randint(20, 245))
        rows = (np.arange(SCREEN[1], dtype=np.float32) / (SCREEN[1] - 1)).reshape(-1, 1, 1)
        bg = np.repeat((np.array(c0, np.float32) * (1 - rows) +
                        np.array(c1, np.float32) * rows), SCREEN[0], axis=1)
    else:
        c0 = (rng.randint(24, 230),) * 3 if rng.random() < 0.5 else \
             (rng.randint(24, 230), rng.randint(24, 230), rng.randint(24, 230))
        bg = np.repeat(np.array(c0, np.float32).reshape(1, 1, 3), SCREEN[1], axis=0)
        bg = np.repeat(bg, SCREEN[0], axis=1)
    out = bg.astype(np.uint8).copy()
    for _ in range(rng.randint(0, 5)):
        ow = rng.randint(SCREEN[0] // 8, SCREEN[0] * 3 // 5)
        oh = rng.randint(SCREEN[1] // 14, SCREEN[1] // 4)
        ox = rng.randint(-ow // 4, max(0, SCREEN[0] - ow // 2))
        oy = rng.randint(0, max(0, SCREEN[1] - oh))
        col = tuple(rng.randint(24, 235) for _ in range(3))
        cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), col, -1)
        cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), (120, 120, 120), 2)
    if rng.random() < 0.20:  # 非棋盘比例细网格（迷惑项）
        gx, gy = rng.randint(3, 7), rng.randint(4, 9)
        x0 = rng.randint(0, SCREEN[0] // 4)
        y0 = rng.randint(0, SCREEN[1] // 4)
        cw = (SCREEN[0] - x0) // gx
        ch = (SCREEN[1] - y0) // gy
        col = tuple(rng.randint(60, 200) for _ in range(3))
        for i in range(gx + 1):
            cv2.line(out, (x0 + i * cw, y0), (x0 + i * cw, y0 + gy * ch), col, 2)
        for j in range(gy + 1):
            cv2.line(out, (x0, y0 + j * ch), (x0 + gx * cw, y0 + j * ch), col, 2)
    return out


# ---------------- 端上同构输入构造 ----------------

def prep_input(canvas_bgr, quad_tl_tr_bl_br, rng=None):
    """整屏 BGR + 角点（模型序 [TL,TR,BL,BR] 原图坐标，None=负样本）
    → (x [3,256,256] float32 已归一化, pts [4,2] float32 256 空间 或 None)。

    与端上链路逐项同构：corner_affine（长边方形窗，v64 新语义——全帧必然覆盖，
    旧 1.25 宽窗在竖屏大棋盘下必然半盘出窗）→ warpAffine 256×256 越界补黑 →
    RGB/255 → ImageNet mean/std。rng 给定时施加光度增广（256 域）。
    """
    h, w = canvas_bgr.shape[:2]
    M = corner_affine(w, h)
    img = cv2.warpAffine(canvas_bgr, M, (CORNER_IN, CORNER_IN),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if rng is not None:
        img = _augment256(img, rng)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb.transpose(2, 0, 1) - IMAGENET_MEAN.reshape(3, 1, 1)) / IMAGENET_STD.reshape(3, 1, 1)
    if quad_tl_tr_bl_br is None:
        return np.ascontiguousarray(x, np.float32), None
    q = np.asarray(quad_tl_tr_bl_br, np.float32) @ M[:, :2].T + M[:, 2]
    return np.ascontiguousarray(x, np.float32), q.astype(np.float32)


def _augment256(img_bgr, rng):
    """256 域光度增广：亮度/对比度/色彩（PIL）→ 噪声/轻模糊（小概率）。"""
    from PIL import Image, ImageEnhance
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.72, 1.32))
    pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.75, 1.30))
    pil = ImageEnhance.Color(pil).enhance(rng.uniform(0.6, 1.35))
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if rng.random() < 0.08:
        arr = arr.astype(np.float32) + np.random.RandomState(rng.randrange(1 << 30)).normal(
            0, rng.uniform(2, 6), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if rng.random() < 0.06:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    return arr


# ---------------- 真实帧池（第 1 波：RTMPose 伪标签） ----------------

class RealPool:
    """伪标签真实帧池：npz 含 files[]（原始帧相对路径）+ corners[N,4,2]（模型序原图坐标）。

    生成脚本 pseudo_label_corner.py 负责 RTMPose 现跑 + boxPlausible 过滤 +
    评测源底片排除清单（eval_photo_*/eval_bench 的 files 全量对照）。
    """

    def __init__(self, npz_path, root=".", split="train"):
        d = np.load(npz_path, allow_pickle=True)
        if "split" in d.files and split:
            keep = d["split"] == split
            self.files = [str(f) for f, k in zip(d["files"], keep) if k]
            self.corners = d["corners"][keep]
        else:
            self.files = [str(f) for f in d["files"]]
            self.corners = d["corners"]
        self.root = Path(root)
        self._cache = {}

    def __len__(self):
        return len(self.files)

    def sample(self, rng):
        i = rng.randrange(len(self.files))
        if i not in self._cache:
            self._cache.clear()  # 常驻 ≤1 张原图（1080×2280×3 ≈ 7MB），防内存堆积
            self._cache[i] = imread_cn(self.root / self.files[i])
        return self._cache[i], self.corners[i]


# ---------------- 混合批供给器 ----------------

class CornerFeeder:
    """线程化预取：渲染整屏（重，一盘 flat 多贴摊薄）+ 真实池（可选）+ 负样本混合出批。

    源配比：neg_ratio（默认 8%）/ real_ratio（默认 30%，池缺失时自动归零给渲染）。
    出批 (x [B,3,256,256] float32, pts [B,4,2] float32 负样本填 0, valid [B] float32)。
    """

    def __init__(self, seed, batch_size, neg_ratio=0.08, real_npz=None,
                 real_ratio=0.30, workers=4):
        self.rng = random.Random(seed)
        self.bs = batch_size
        self.fonts = [f for f in R.load_font_pool()
                      if f.split("\\")[-1].split("/")[-1] not in GEN_HOLDOUT]
        self.pool = RealPool(real_npz) if real_npz else None
        if self.pool is None or len(self.pool) == 0:
            self.pool = None
            real_ratio = 0.0
        self.neg_ratio = neg_ratio
        self.real_ratio = real_ratio
        self.q = queue.Queue(maxsize=6)
        self.stop = False
        self.threads = [threading.Thread(target=self._fill, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def _render_one(self, wr):
        """渲染源样本：一盘 flat 复用 3~6 次（几何/背景/遮挡逐帧重采）。"""
        if getattr(wr, "_flat", None) is None or wr._uses <= 0:
            lay = R.random_layout(wr, wr.choices(["opening", "midgame", "endgame"],
                                                  weights=[1, 2, 1])[0])
            style = R.sample_style(wr, self.fonts)
            wr._flat = R.render_board_flat(lay, style, wr)
            wr._uses = wr.randint(8, 14)
        wr._uses -= 1
        canvas, quad = corner_place(wr._flat, wr)     # [TL,TR,BR,BL]
        q_model = np.array([quad[0], quad[1], quad[3], quad[2]], np.float32)  # → TL,TR,BL,BR
        return prep_input(canvas, q_model, rng=wr)

    def _one_sample(self, wr):
        u = wr.random()
        if u < self.neg_ratio:
            return prep_input(make_negative_frame(wr), None, rng=wr)
        if self.pool is not None and u < self.neg_ratio + self.real_ratio:
            img, cor = self.pool.sample(wr)
            return prep_input(img, cor, rng=wr)
        return self._render_one(wr)

    def _fill(self):
        wr = random.Random(self.rng.randrange(1 << 30))
        while not self.stop:
            xs, pts, valid = [], [], []
            for _ in range(self.bs):
                x, p = self._one_sample(wr)
                xs.append(x)
                pts.append(p if p is not None else np.zeros((NKP, 2), np.float32))
                valid.append(0.0 if p is None else 1.0)
            batch = (torch.from_numpy(np.stack(xs)),
                     torch.from_numpy(np.stack(pts)).float(),
                     torch.tensor(valid, dtype=torch.float32))
            self.q.put(batch, timeout=300)

    def next(self):
        return self.q.get(timeout=300)

    def close(self):
        self.stop = True


def make_val(seed, n):
    """固定 seed 渲 n 帧干净整屏（无增广）→ (x [n,3,256,256], pts [n,4,2])，可复现。"""
    rng = random.Random(seed)
    fonts = [f for f in R.load_font_pool()
             if f.split("\\")[-1].split("/")[-1] not in GEN_HOLDOUT]
    xs, pts = [], []
    while len(xs) < n:
        lay = R.random_layout(rng, rng.choice(["opening", "midgame", "endgame"]))
        style = R.sample_style(rng, fonts)
        flat = R.render_board_flat(lay, style, rng)
        for _ in range(3):
            if len(xs) >= n:
                break
            canvas, quad = corner_place(flat, rng)
            q_model = np.array([quad[0], quad[1], quad[3], quad[2]], np.float32)
            x, p = prep_input(canvas, q_model)
            xs.append(x)
            pts.append(p)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(pts)).float())


if __name__ == "__main__":
    # 冒烟自测：渲染/负样本各 1 帧出形状 + 可视化留档
    rng = random.Random(7)
    feeder_probe = CornerFeeder(11, 2, workers=0)  # workers=0：手动单线程探针
    x, p = feeder_probe._render_one(rng)
    print("渲染样本:", x.shape, x.dtype, "pts:", None if p is None else p.round(1).tolist())
    xn, pn = prep_input(make_negative_frame(rng), None)
    print("负样本:", xn.shape, "pts:", pn)
    cv2.imwrite("../../.cache/corner_smoke_render.png",
                ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8).transpose(1, 2, 0)[..., ::-1])
    print("已存 .cache/corner_smoke_render.png（归一化域可视化）")
