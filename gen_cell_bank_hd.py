# -*- coding: utf-8 -*-
"""v2 数据：real 源图 HD warp 格 bank（原生高清真实域，cell_cls HD 战力来源）。

源图（v24 360 + v47 1023 + real_frames 266 = 1649 张）→ RTMPose 角点 → HD warp
（900×1000，格 87.5px 原生）→ r3 打标（resize 回 315×280 推理，pmax>0.85 才入库）
→ 每图抽 30 格 88px→64px u8 存储。

产出 data/cell_bank_hd.npz（cells[N,64,64,3] + labels[N] + src[N]）。
注意：与 26 帧评测集无交集（不同源图池；评测帧早已从训练源图体系隔离）。
"""
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

import eval_cascade as EC
import pose_corner as PC

HERE = Path(__file__).resolve().parent
SRC_DIRS = [("v24", HERE / "data/photo_crawl_v24/raw"),
            ("v47", HERE / "data/photo_crawl_v47/raw"),
            ("rf", HERE / "data/real_frames")]
GRID_DST_HD = np.array([[100, 100], [800, 100], [100, 900], [800, 900]], np.float32)
PER_IMG = 30
CONF_THR = 0.85
rng = random.Random(410)


def cls_view_of_hd(hd900: np.ndarray) -> np.ndarray:
    """HD warp 900×1000 BGR → 315×280 RGB（与评测链同几何：网格区居中裁剪+缩放）。"""
    crop = hd900[50:950, 50:850]                     # 网格区 (100,100)-(800,900) 外扩 50
    img = cv2.resize(crop, (280, 315), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    det = PC.PoseCornerDetector()
    sess_r3 = ort.InferenceSession(EC.R3, providers=["CPUExecutionProvider"])
    all_c, all_l, all_s = [], [], []
    nfail = 0
    for tag, d in SRC_DIRS:
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpeg"))
        for p in files:
            src = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if src is None:
                continue
            if max(src.shape[:2]) > 1600:
                src = cv2.resize(src, (1600, int(src.shape[0] * 1600 / src.shape[1])))
            corners = det.detect_with_check(src)
            if corners is None:
                nfail += 1
                continue
            H = cv2.getPerspectiveTransform(corners.astype(np.float32), GRID_DST_HD)
            hd = cv2.warpPerspective(src, H, (900, 1000), flags=cv2.INTER_LINEAR)
            out = EC.infer_r3(sess_r3, cls_view_of_hd(hd))   # r3 打标（真实域 99.5% 准）
            ok = (out["pmax"] > CONF_THR) & (out["top1"] >= 0)
            cand = [i for i in range(90) if ok[i]]
            if not cand:
                nfail += 1
                continue
            rng.shuffle(cand)
            for i in cand[:PER_IMG]:
                r, c = divmod(i, 9)
                cx, cy = 100 + 87.5 * c, 100 + 88.888 * r
                crop = hd[int(cy - 44):int(cy + 44), int(cx - 44):int(cx + 44)]
                cell64 = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                all_c.append(cv2.cvtColor(cell64, cv2.COLOR_BGR2RGB))
                all_l.append(int(out["top1"][i]))
                all_s.append(tag)
        print(f"[{tag}] 完成，累计 {len(all_c)} 格（角点/打标失败 {nfail}）", flush=True)
    cells = np.stack(all_c)
    np.savez_compressed(HERE / "data/cell_bank_hd.npz", cells=cells,
                        labels=np.array(all_l, np.int64), src=np.array(all_s))
    u, cnt = np.unique(all_l, return_counts=True)
    print(f"产出 data/cell_bank_hd.npz: {cells.shape}, 失败 {nfail}, "
          f"类分布 {dict(zip(u.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()
