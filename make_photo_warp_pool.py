# -*- coding: utf-8 -*-
"""街拍 warp 池工厂（v66 r3 整板重训弹药）：photo 双闸清洗池 → nano 伪标签 + 编制闸 + 一致格 IGNORE。

管线与部署逐位同构：imread_cn → 角点 warp450（probe_nano_pseudo 同几何）→ 中心裁 400×450
→ 缩 280×315 RGB。标签源 = nano argmax；一致性伙伴 = r3 级联 p3（现役 w085 onnx）。
全程 raw 朝向（不 canonicalize）——部署时 r3 看到的就是原始朝向（canonical 在预测后处理），
与 annotate_real 老池约定一致。

两道闸（探针 probe_nano_pseudo 预注册判定双过后的落地）：
  闸1 编制校验（帧级）：车≤2马≤2炮≤2兵≤5仕≤2相≤2、帅将各恰 1——失败整帧弃
  闸2 一致格 IGNORE（格级）：nano=='x'（教师叫不出名）或 nano≠r3 的格标 -100，
      训练侧 CE ignore_index 自动跳过（v3 先例：8020 格 IGNORE 同机制）

输出 data/real_photo_warp_v66.npz：imgs [N,315,280,3] u8 RGB NHWC + labels [N,90] i64
+ files（审计字段，train.py 不读）。红线：输出文件名对 corner_excl.json 358 评测源
（26帧/bench/gold/photo_crawl_eval 全在内）逐一断言零交集，违例拒存。
用法：python make_photo_warp_pool.py（cwd 任意）
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_screen30 import (       # noqa: E402  现役 r3/nano 会话与部署管线函数（模块级加载）
    MEAN, STD, sess_nano, cls_input_280x315, r3_probs, cell_out, full_cascade,
)
from probe_nano_pseudo import composition_ok, warp_by_corners   # noqa: E402  探针同款闸与几何
from corner_data import imread_cn                               # noqa: E402  中文路径安全读图

CRAWL = HERE                     # files 字段相对 recog（data\photo_crawl\raw\…）
POOL = HERE / "../../.cache/corner_photo_clean.npz"
EXCL = HERE / "../../.cache/corner_excl.json"
OUT = HERE / "data/real_photo_warp_v66.npz"


def main():
    d = np.load(str(POOL), allow_pickle=True)
    files, corners = d["files"], d["corners"]
    excl = {str(n).replace("\\", "/").split("/")[-1]
            for n in json.loads(EXCL.read_text(encoding="utf-8"))["exclude_names"]}

    imgs, labs, kept_files = [], [], []
    stats = Counter()
    ign_reason = Counter()
    cls_hist = Counter()
    for f, gt_c in zip(files, corners):
        name = str(f).replace("\\", "/").split("/")[-1]
        assert name not in excl, f"红线违例：{name} 在评测排除清单内"
        img = imread_cn(CRAWL / str(f))
        if img is None:
            stats["read_fail"] += 1
            continue
        img280 = cls_input_280x315(warp_by_corners(img, np.asarray(gt_c, np.float32)))
        x = (img280.astype(np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
        lo = sess_nano.run(None, {sess_nano.get_inputs()[0].name:
                                  np.ascontiguousarray(x[None])})[0][0]      # [90,15]
        npred = lo.argmax(1)
        if not composition_ok(npred):          # 闸1：编制校验（帧级，含帅将各恰 1）
            stats["comp_fail"] += 1
            continue
        o = r3_probs(img280)
        cp, cc = cell_out(img280)
        p3, _ = full_cascade(o["top1"], o["pmax"], o["second"], cp, cc)   # r3 级联（一致性伙伴）
        lab = npred.astype(np.int64).copy()
        m_x = npred == 1                       # 闸2：'x'=教师叫不出名
        m_dis = npred != p3                    # 闸2：nano/r3 不一致格
        lab[m_x | m_dis] = -100
        ign_reason["nano_x"] += int(m_x.sum())
        ign_reason["disagree"] += int(m_dis.sum())
        for c in lab[lab >= 0]:
            cls_hist[int(c)] += 1
        stats["kept"] += 1
        stats["ign_cells"] += int((lab == -100).sum())
        imgs.append(img280)
        labs.append(lab)
        kept_files.append(str(f))

    n = stats["kept"]
    arr = np.stack(imgs)                        # [N,315,280,3] u8 RGB
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT), imgs=arr, labels=np.stack(labs).astype(np.int64),
             files=np.array(kept_files))
    out_names = {str(f).replace("\\", "/").split("/")[-1] for f in kept_files}
    assert not (out_names & excl), "红线断言失败：输出池与评测排除清单有交集，拒收"

    taught = n * 90 - stats["ign_cells"]
    print(f"== 街拍 warp 池工厂 ==")
    print(f"读入 {len(files)} | 读失败 {stats['read_fail']} | 编制闸弃 {stats['comp_fail']} "
          f"| 保留 {n} 帧")
    print(f"IGNORE 格 {stats['ign_cells']}/{n*90} = {stats['ign_cells']/(n*90):.3f} "
          f"（nano_x {ign_reason['nano_x']} + 不一致 {ign_reason['disagree']}）"
          f"；实教格 {taught}")
    occ = sum(v for k, v in cls_hist.items() if k not in (0, 1))
    print(f"类别分布：空格 {cls_hist.get(0,0)} 占子 {occ} "
          f"（占子率 {occ/max(taught,1):.3f}）")
    print(f"输出 {OUT}（{arr.nbytes/2**20:.0f}MB）+ 红线断言通过（358 评测源零交集）")


if __name__ == "__main__":
    main()
