# -*- coding: utf-8 -*-
"""roboflow COCO 标注 → 真实域监督训练池（v42 主攻，docs/v42真实域监督数据方案-20261017.md）。

把 detection_14class 的 17330 张真实整图 + COCO 人工棋子框标注，经角点对齐产出
315×280 分类帧 + 90 格标签——**标签完全来自人工标注，不经 nano 之手**（绕开蒸馏天花板）。

管线：COCO 解析 → 类别映射 → RTMPose 角点对齐 → 框中心→最近格点 → 结构校验 → npz。
质量门：角点成功/框-格距离 <0.5 格距/结构合法（帅将各一/子数上限）。
越南语系项目（chess-abplc/chinese-chess-fpt/robot-pi）跳过：tuong/tuongx 语义互换历史坑，
核验后另行接入。'chess'/'objects' 等不分种类的框丢弃。

用法：
  python parse_roboflow_coco.py --out data/real_roboflow_sup.npz
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pose_corner import PoseCornerDetector, warp_frame_to_warp450

# COCO 类名 → 项目 16 类（.xKABNRCPkabnrcp）。
# 两套英文命名并存：general/advisor/chariot/elephant/horse 系与 king/bishop/knight/rook 系
# （同义词归并：rook=chariot=车、bishop=elephant=象、knight=horse=马——映射无冲突）
NAME2CLS = {
    # 英文系 A：general/advisor/chariot/elephant/horse
    "black_general": "k", "black_advisor": "a", "black_elephant": "b", "black_horse": "n",
    "black_chariot": "r", "black_cannon": "c", "black_soldier": "p",
    "red_general": "K", "red_advisor": "A", "red_elephant": "B", "red_horse": "N",
    "red_chariot": "R", "red_cannon": "C", "red_soldier": "P",
    # 英文系 B：king/bishop/knight/rook/pawn
    "black_king": "k", "black_bishop": "b", "black_knight": "n",
    "black_rook": "r", "black_pawn": "p",
    "red_king": "K", "red_bishop": "B", "red_knight": "N",
    "red_rook": "R", "red_pawn": "P",
    # 拼音系（b_=黑 r_=红）
    "b_jiang": "k", "b_shi": "a", "b_xiang": "b", "b_ma": "n",
    "b_che": "r", "b_pao": "c", "b_zu": "p",
    "r_shuai": "K", "r_shi": "A", "r_xiang": "B", "r_ma": "N",
    "r_che": "R", "r_pao": "C", "r_bing": "P",
}
CLS_CHARS = ".xKABNRCPkabnrcp"
CLS_INDEX = {ch: i for i, ch in enumerate(CLS_CHARS)}

# 越南语系三集映射（2026-10-18 框中心坐标分布核验定案，docs/v44部署修复与实物扩量方案-20261017.md §2.1；
# 当年"tuong 语义互换坑"是下载时未验证假设，本轮用坐标分布+框数比完成核验）：
# - abplc/fpt：vua=王(tướng)/tuong=象(tượng)——vua mean|nx-0.5|=0.125 与王理论值精确吻合；
# - robot-pi：tuong=将/tuongx=象，与调研文档假设**正好相反**——框数比 tuongx:tuong=2.02:1
#   精确等于象:将规则比。反向映射残余风险由两道安全网兜底：struct_ok 帅将各一 +
#   贴片管线格级自证（nano 原帧认对才入库，标反格自动被拒）。
# 颜色后缀：_do=đỏ 红 / _den=đen 黑。robot-pi 实测带颜色后缀（此前"无颜色"判断有误）。
VN_COLOR = {"ma_den": "n", "ma_do": "N", "phao_den": "c", "phao_do": "C",
            "si_den": "a", "si_do": "A", "tot_den": "p", "tot_do": "P",
            "xe_den": "r", "xe_do": "R"}
VN_MAPS = {
    "chess-abplc": {"vua_do": "K", "vua_den": "k", "tuong_do": "B", "tuong_den": "b", **VN_COLOR},
    "chinese-chess-fpt": {"vua_do": "K", "vua_den": "k", "tuong_do": "B", "tuong_den": "b", **VN_COLOR},
    "robot-pi-j9a5f": {"tuong_do": "K", "tuong_den": "k", "tuongx_do": "B", "tuongx_den": "b", **VN_COLOR},
}
WARP_W, WARP_H = 450, 500
# 角点顺序 TL,TR,BL,BR（与 pose_corner.warp_frame_to_warp450 的 GRID_DST 一致）
GRID_DST = np.array([[50, 50], [400, 50], [50, 450], [400, 450]], np.float32)
CELL_X = np.array([50 + (400 - 100) * c / 8 for c in range(9)])
CELL_Y = np.array([50 + (450 - 100) * r / 9 for r in range(10)])
HALF_CELL = 30.0            # 框-格距离阈值：argmin 最近格归属保证不跳格（次近格必更远），
                            # 30px 容纳 RTMPose 角点误差；22px 半格距口径下入池率仅 20%
PIECE_LIMITS = {"K": 1, "k": 1, "A": 2, "a": 2, "B": 2, "b": 2,
                "N": 2, "n": 2, "R": 2, "r": 2, "C": 2, "c": 2, "P": 5, "p": 5}


def struct_ok(labels90: np.ndarray) -> bool:
    """结构校验：帅/将各恰一 + 子型上限 + 总子数 ≤32（只数占子格）。
    注意 v>=0 会把空格 '.'=0 计入导致总数恒 90——v37 时代 audit_labels.py 同款坑。"""
    cnt = Counter(CLS_CHARS[v] for v in labels90 if v > 0)
    if cnt.get("K", 0) != 1 or cnt.get("k", 0) != 1:
        return False
    if sum(cnt.values()) > 32:
        return False
    return all(cnt.get(ch, 0) <= lim for ch, lim in PIECE_LIMITS.items())


def process_image(detector, img_bgr: np.ndarray,
                  boxes: list[tuple[float, float, float, float, int]],
                  stat: Counter | None = None):
    """单图处理：角点对齐 → 315×280 帧 + 90 格标签 + 框心偏移。返回 (cls_img, labels, offsets) 或 None。
    boxes: [(cx, cy, w, h, cls_idx)] 原图坐标的框中心+尺寸+类别索引。
    offsets: [90,2] float32——占子格存「COCO 框心 − 真实格心」在 cls 域(280×315)的偏移
    （v43 贴片按框心裁剪根治 σ≈10-13px 偏心裁剪，docs/v43实物贴片方案-20261017.md §2.2）；
    注意参考系是校正前的原始格点（真实渲染几何），而非落格用的平移校正网格。"""
    corners = detector.detect_with_check(img_bgr)
    if corners is None:
        if stat is not None:
            stat["角点失败"] += 1
        return None
    warp = warp_frame_to_warp450(img_bgr, corners)
    # 端上同款：裁 25:475/25:425 → 缩 280×315（RGB）
    cls_img = cv2.cvtColor(cv2.resize(warp[25:475, 25:425], (280, 315)),
                           cv2.COLOR_BGR2RGB)
    labels = np.full(90, CLS_INDEX["."], np.int64)
    offsets = np.zeros((90, 2), np.float32)
    # 框中心 → warp450 坐标 → 最近格点；两轮落格：
    # 第一轮收集阈内框的（格心-框心）平均偏移——RTMPose 存在 ~(+18,+14)px 的一致性
    # 平移偏移（docs 记录），最小二乘平移校正后重新落格，远框大批 rescued；
    # 同格冲突按距离保留近者（丢框 >3 说明对齐烂、整帧降级）
    if boxes:
        Hm = cv2.getPerspectiveTransform(corners.astype(np.float32), GRID_DST)
        pts = np.array([[b[0], b[1]] for b in boxes], np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)
        grid_pts = np.array([[x, y] for y in CELL_Y for x in CELL_X], np.float32)  # [90,2]
        grid_pts_orig = grid_pts.copy()        # 偏移参考系：校正前的真实渲染格心
        for _round in range(2):
            cand = []
            for (cx_w, cy_w), (_, _, _, _, ci) in zip(warped, boxes):
                d = np.hypot(grid_pts[:, 0] - cx_w, grid_pts[:, 1] - cy_w)
                k = int(d.argmin())
                if d[k] < HALF_CELL:
                    cand.append((float(d[k]), k, ci, cx_w, cy_w))
            if _round == 0 and len(cand) >= 6:
                dx = float(np.mean([c[3] - grid_pts_orig[c[1], 0] for c in cand]))
                dy = float(np.mean([c[4] - grid_pts_orig[c[1], 1] for c in cand]))
                if abs(dx) > 2 or abs(dy) > 2:
                    grid_pts = grid_pts + np.array([dx, dy], np.float32)
                    continue                      # 校正后第二轮重新落格
            break
        cand.sort(key=lambda c: c[0])
        occupied: dict[int, tuple[int, float, float]] = {}
        dropped = 0
        for _, k, ci, cx_w, cy_w in cand:
            if k in occupied:
                dropped += 1                     # 同格已有更近的框：丢弃后来者
                continue
            occupied[k] = (ci, cx_w, cy_w)
            labels[k] = ci
            # warp450→cls 域只有平移(-25)与缩放(×0.7)，差值不受平移影响
            offsets[k] = ((cx_w - grid_pts_orig[k, 0]) * 0.7,
                          (cy_w - grid_pts_orig[k, 1]) * 0.7)
        if dropped > 3:
            if stat is not None:
                stat["丢框过多"] += 1
            return None
    if not struct_ok(labels):
        if stat is not None:
            stat["结构不合法"] += 1
        return None
    return cls_img, labels, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).parent.parent.parent.parent
                                           / "data/roboflow_raw/detection_14class"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "data/real_roboflow_sup.npz"))
    ap.add_argument("--min-boxes", type=int, default=6, help="图内有效框下限（少于则跳过）")
    ap.add_argument("--limit", type=int, default=0, help="调试：限制处理图数")
    args = ap.parse_args()

    root = Path(args.root)
    out_imgs, out_labs, out_src, out_off = [], [], [], []
    stat = Counter()
    t0 = time.time()
    detector = PoseCornerDetector()

    jsons = sorted(root.glob("**/_annotations.coco.json"))
    print(f"发现 {len(jsons)} 个 COCO json", flush=True)
    n_done = 0
    for jp in jsons:
        split_dir = jp.parent
        d = json.load(open(jp, encoding="utf-8"))
        proj = jp.relative_to(root).parts[0]
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        # 类别映射（本 json 内）：越南语系三集走条件映射，其余走全局 NAME2CLS
        cat2cls = {}
        if proj in VN_MAPS:
            vn = VN_MAPS[proj]
            for cid, name in cats.items():
                if name in vn:
                    cat2cls[cid] = CLS_INDEX[vn[name]]
        else:
            for cid, name in cats.items():
                if name in NAME2CLS:
                    cat2cls[cid] = CLS_INDEX[NAME2CLS[name]]
        anns_by_img = {}
        for a in d.get("annotations", []):
            if a["category_id"] in cat2cls:
                anns_by_img.setdefault(a["image_id"], []).append(a)
        id2file = {im["id"]: im["file_name"] for im in d.get("images", [])}
        for img_id, fname in id2file.items():
            n_done += 1
            if args.limit and n_done > args.limit:
                break
            boxes_raw = anns_by_img.get(img_id, [])
            fpath = split_dir / fname
            if not fpath.exists():
                stat["图缺失"] += 1
                continue
            img = cv2.imread(str(fpath))
            if img is None:
                stat["读取失败"] += 1
                continue
            boxes = []
            for a in boxes_raw:
                bx, by, bw, bh = a["bbox"]
                boxes.append((bx + bw / 2, by + bh / 2, bw, bh, cat2cls[a["category_id"]]))
            if len(boxes) < args.min_boxes:
                stat["框不足"] += 1
                continue
            r = process_image(detector, img, boxes, stat)
            if r is None:
                continue
            cls_img, labels, offs = r
            out_imgs.append(cls_img)
            out_labs.append(labels)
            out_src.append(proj)
            out_off.append(offs)
            stat["入池"] += 1
        if args.limit and n_done > args.limit:
            break
        if n_done % 2000 < len(id2file):
            print(f"  进度 {n_done} 图，入池 {stat['入池']}，{time.time()-t0:.0f}s", flush=True)

    print(f"\n== 统计 ==\n{dict(stat)}")
    if out_imgs:
        imgs = np.stack(out_imgs)
        labs = np.stack(out_labs)
        offs = np.stack(out_off)
        np.savez(args.out, imgs=imgs, labels=labs, offsets=offs,
                 src=np.array(out_src), n_frames=len(imgs))
        # 类别分布
        occ = labs[labs >= 0]
        dist = Counter(CLS_CHARS[v] for v in occ)
        print(f"入池 {len(imgs)} 帧（{imgs.nbytes/2**20:.0f}MB）→ {args.out}")
        print(f"占子格类别分布: {dict(sorted(dist.items()))}")
        oc = offs[np.linalg.norm(offs, axis=2) > 0.5]
        if len(oc):
            print(f"框心偏移: 占子格 {len(oc)} 个，σ=({oc[:,0].std():.1f},{oc[:,1].std():.1f})px，"
                  f"均值=({oc[:,0].mean():.1f},{oc[:,1].mean():.1f})px")


if __name__ == "__main__":
    main()
