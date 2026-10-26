# -*- coding: utf-8 -*-
"""v67 QQ 事故截图基准对拍（判据：定位全对；不入库临时脚本）。

v67 box 模型检测四角 vs corner_qq_basis.npz 的 v4 GT（目检+全链路 89/90 实证）：
四角平均距离 ≤6px@原图 判 PASS（6px ≈ 0.4% 板宽，远小于切格阈值）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent          # recog 目录
CACHE = HERE / "data"                                      # 数据缓存根（不入库）
sys.path.insert(0, str(HERE))
from corner_infer import CornerInfer  # noqa: E402

PT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "runs/corner_v67/best.pt")


def main():
    d = np.load(str(CACHE / "corner_qq_basis.npz"), allow_pickle=True)
    img = cv2.imdecode(np.fromfile(str(d["files"][0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    gt = d["corners"][0]
    inf = CornerInfer(torch_weights=Path(PT))
    det = inf.detect_with_check(img)
    if det is None:
        print("v67 QQ 基准: detect_with_check 返回 None（护栏拒绝）→ FAIL")
        return
    dist = np.linalg.norm(det - gt, axis=1)
    print(f"v67 四角: {det.astype(int).tolist()}")
    print(f"v4 GT : {gt.astype(int).tolist()}")
    print(f"四角距离 px: {dist.round(1).tolist()}  平均 {dist.mean():.1f}")
    print(f"v67 QQ 基准: {'PASS' if dist.mean() <= 6.0 else 'FAIL'}（判线 ≤6px 平均）")
    vis = img.copy()
    cv2.polylines(vis, [gt.astype(np.int32)], True, (255, 0, 0), 5)      # 蓝=v4 GT
    cv2.polylines(vis, [det.astype(np.int32)], True, (0, 0, 255), 3)     # 红=v67
    cv2.imwrite(str(CACHE / "qq_v67_check.png"), cv2.resize(vis, None, fx=0.4, fy=0.4))


if __name__ == "__main__":
    main()
