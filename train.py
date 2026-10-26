# -*- coding: utf-8 -*-
"""训练脚本（自研识别引擎一期，docs/自研识别引擎方针.md 阶段2）。

数据方针：在线渲染（renderer.make_sample 每次现做），无限样本多样性，
无需落盘数据集；验证集启动时用固定种子渲染（可复现、不参与训练）。

字体池纪律：泛化评测用的 held-out 字体（GEN_HOLDOUT）不进训练池——
它是「没见过的字体风格」这一泛化维度的唯一可信来源。

运行（4060 约 1~2 小时量级）：
  python train.py                       # 默认超参，断点续训
  python train.py --max-minutes 9      # 时间盒模式（前台分片执行用）
产物：runs/<name>/best.pt（按验证 full-err-0 选优）、ckpt.pt（含优化器状态）、log.txt
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import renderer as R
from model import GridHead, NCLASS, ROWS, COLS

# 泛化评测专用字体（训练时排除；build_eval.py 生成泛化集时只用它们）
GEN_HOLDOUT = ["NotoSerifSC-Regular.ttf", "ZCOOLXiaoWei-Regular.ttf", "SourceHanSerifCN-Bold.otf"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# numpy 版（CPU 推理路径用；历史蒸馏生成器已随路线归档移除）
IM_MEAN_NP = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IM_STD_NP = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def str2bool(v: str) -> bool:
    """pipeline 把 preset 的布尔键透传成 `--attn True` 字符串形式，argparse store_true
    接不住，统一用本解析器。"""
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def font_pools():
    """返回 (训练字体池, 泛化字体池)。泛化池 = 全池 ∩ GEN_HOLDOUT。"""
    all_fonts = R.load_font_pool()
    train = [f for f in all_fonts if Path(f).name not in GEN_HOLDOUT]
    gen = [f for f in all_fonts if Path(f).name in GEN_HOLDOUT]
    return train, gen


class RenderDataset(Dataset):
    """整批渲染数据集：worker 进程内一次渲染一个完整 batch 再返回。
    逐样本返回时 Windows IPC 序列化开销会让吞吐减半（实测 95 vs 预期 390 样本/s），
    批内渲染把 IPC 摊薄为每批一次。播种方案与逐样本版等价（idx*B+j）。

    蒸馏样本不经过本数据集：npz 是 zip 容器（mmap 无效，每次访问重新解压），
    且 Windows spawn 下 3.2GB 数组会被每个 worker 复制一份。改由主进程一次性
    载入内存，在主循环批尾槽替换（见 main）。"""

    def __init__(self, base_seed: int, fonts: list, batch_size: int, length_batches: int = 10 ** 7):
        self.base_seed = base_seed
        self.fonts = fonts
        self.batch = batch_size
        self.length = length_batches

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        B = self.batch
        imgs = torch.empty((B, 3, R.CLS_IN_H, R.CLS_IN_W), dtype=torch.uint8)
        labs = torch.empty((B, R.ROWS * R.COLS), dtype=torch.int64)
        for j in range(B):
            rng = random.Random((self.base_seed * 1_000_003 + idx * B + j) % (2 ** 63))
            s = R.make_sample(rng, self.fonts)
            imgs[j] = torch.from_numpy(np.ascontiguousarray(s["cls_img"].transpose(2, 0, 1)))
            labs[j] = torch.from_numpy(s["labels"])
        return imgs, labs


def normalize_gpu(batch_u8: torch.Tensor) -> torch.Tensor:
    """GPU 侧 ImageNet 归一化（与 OnnxGeom.normalizeChwInto 等价：/255 + mean/std, RGB）"""
    x = batch_u8.float().div_(255.0)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def quantize_weights_(model: torch.nn.Module) -> None:
    """QAT 步前处理：把全部 Conv/Linear 权重 snap 到 int8 per-channel 对称网格
    （scale=amax|w|/127，与 export.py 的 weight_type=QInt8 per_channel=True 对齐）。
    梯度更新把权重推离网格、下一步重新 snap——权重始终在量化值域上优化。"""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight
                d = list(range(1, w.dim()))
                s = w.abs().amax(dim=d, keepdim=True).clamp_min(1e-8) / 127.0
                w.copy_((w / s).round().clamp(-127, 127) * s)


class ActFakeQuant:
    """QAT-v2 激活 fake-quant（STE）：对 Conv/Linear 输出模拟 QDQ QUInt8 非对称
    per-tensor 量化——scale=(amax-amin)/255（与导出端 MinMax 校准对齐），
    forward=量化值、backward 梯度直通。v43q 实测：权重 QAT 后 int8 bench 4/10 vs
    网格 fp32 7/10，gap 全为激活量化噪声——此模块补另一半。

    热身机制（v43q2 首跑崩坏教训）：构造后 enabled=False，仅以 0.5 动量快速收敛
    EMA range（不量化）；外部在 range 稳定后（如 step≥200）置 enabled=True——
    冷启动即量化会让未收敛 range 的裁剪噪声带崩训练与全部选点（step 50 bench
    0/10、gen 5 倍劣化）。启用后 momentum 0.9 慢速跟踪。选点/val 时 hook 生效
    =选点口径即部署口径模拟（range 固定后，范围外激活被裁正是部署真实行为）。"""

    def __init__(self, model: torch.nn.Module, momentum: float = 0.9,
                 warmup_momentum: float = 0.5):
        self.momentum = momentum
        self.warmup_momentum = warmup_momentum
        self.enabled = False
        self.range: dict[int, list] = {}      # id(module) -> [amin, amax]
        self.handles = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.range[id(m)] = [None, None]
                self.handles.append(m.register_forward_hook(self._make(id(m))))

    def _fq(self, out: torch.Tensor, amin: float, amax: float) -> torch.Tensor:
        scale = max(1e-8, (amax - amin) / 255.0)
        zp = -amin / scale
        q = torch.clamp(torch.round(out / scale + zp), 0, 255)
        return out + ((q - zp) * scale - out).detach()   # STE

    def _make(self, key: int):
        def hook(mod, inp, out):
            r = self.range[key]
            with torch.no_grad():
                mn = float(out.min().item()); mx = float(out.max().item())
                if r[0] is None:
                    r[0], r[1] = mn, mx
                else:
                    mom = self.momentum if self.enabled else self.warmup_momentum
                    r[0] = mom * r[0] + (1 - mom) * mn
                    r[1] = mom * r[1] + (1 - mom) * mx
            if not self.enabled:
                return None
            return self._fq(out, r[0], r[1])
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def evaluate(model: GridHead, loader: DataLoader, device: str) -> dict:
    """验证：格子准确率（含/仅占子格）与 full-err-k（整盘错误 ≤k 子的占比）"""
    model.eval()
    n_cell = n_cell_ok = n_occ = n_occ_ok = 0
    board_err: list[int] = []
    for u8, lab in loader:
        u8, lab = u8.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        logits = model(normalize_gpu(u8))                     # [N,90,16]
        pred = logits.argmax(dim=2)                           # [N,90]
        valid = lab != R.IGNORE_LABEL
        occ = valid & (lab != R.CLASS_INDEX["."])
        ok = (pred == lab) & valid
        n_cell += int(valid.sum()); n_cell_ok += int(ok.sum())
        n_occ += int(occ.sum()); n_occ_ok += int((ok & occ).sum())
        board_err += (~ok & valid).sum(dim=1).tolist()        # 每盘错误格数
    be = np.array(board_err)
    return {
        "cell_acc": n_cell_ok / max(1, n_cell),
        "occ_acc": n_occ_ok / max(1, n_occ),
        "full_err0": float((be == 0).mean()),
        "full_err1": float((be <= 1).mean()),
        "full_err2": float((be <= 2).mean()),
        "boards": len(be),
    }


def make_val_loader(seed: int, fonts: list, n: int, workers: int) -> DataLoader:
    """固定种子的验证集：先用线程池预渲染成内存数组（可复现、快）"""
    from concurrent.futures import ThreadPoolExecutor
    import random as _r

    def one(i: int):
        rng = _r.Random(seed * 7_000_001 + i)
        s = R.make_sample(rng, fonts)
        return s["cls_img"].transpose(2, 0, 1).copy(), s["labels"]

    with ThreadPoolExecutor(max_workers=min(16, workers * 2)) as ex:
        items = list(ex.map(one, range(n)))
    imgs = torch.from_numpy(np.stack([it[0] for it in items]))
    labs = torch.from_numpy(np.stack([it[1] for it in items]))
    # batch 96 而非 256：Windows cuDNN 9.10 启发式对 256 形状挑中病态算法
    # （隔离实测：批 256 前向 23.5s=11帧/s，批 96 生产实测 100帧/s），
    # v36 轮 val 周期 173s→预计约 40s；评测按批聚合与 batch 大小无关，数字口径不变
    return DataLoader(torch.utils.data.TensorDataset(imgs, labs), batch_size=96, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--total-samples", type=int, default=400_000, help="总训练样本数（虚拟 epoch=20k）")
    ap.add_argument("--epoch-samples", type=int, default=20_000)
    ap.add_argument("--warmup", type=int, default=400, help="warmup 步数")
    ap.add_argument("--val-every", type=int, default=150, help="每 N 步验证一次")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=0, help=">0 时时间盒模式：到点存盘退出")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--width", default="", help="加宽通道四元组（逗号分隔，如 112,168,256,352）；空=默认 64,96,160,224（132K）")
    ap.add_argument("--head-dim", type=int, default=64, help="头部 1x1 压缩通道数")
    # v3：真实帧混训 + 尺度下探 + 从 v2 best 初始化
    ap.add_argument("--real-npz", default=str(Path(__file__).parent / "data/real_frames_labeled.npz"),
                    help="真实帧标注集（annotate_real.py 产出）；空串=纯合成")
    ap.add_argument("--real-ratio", type=float, default=0.2,
                    help="每 batch 中真实样本占比（0=纯合成）")
    ap.add_argument("--scale-lo", type=float, default=0.94,
                    help="尺度增广下探（v3 用 0.85 覆盖外框错位带）")
    ap.add_argument("--init-from", default="", help="初始权重来源（如 runs/v2/best.pt），仅模型权重")
    # v60 软标签蒸馏：nano logits 挂 real 池帧（--real-npz 按序一一对应），
    # CE(硬标签) + λ·T²·KL(teacher/T ‖ student/T) 仅作用蒸馏帧 valid 格（IGNORE 跳过）。
    # 与 --distill-npz 语义分离（独立批尾槽 vs real 帧双损失），二者互斥。
    ap.add_argument("--distill-soft", default="",
                    help="逗号分隔 soft npz 清单（distill_build.py 产出，与 --real-npz 按序对应）；空=不用")
    ap.add_argument("--distill-T", type=float, default=4.0,
                    help="蒸馏温度（v60 抽检降档：rf 池 teacher 平坦，T=4 压噪）")
    ap.add_argument("--distill-lambda", type=float, default=0.3,
                    help="KL 项权重 λ（抽检一致率 <85% 降档 0.5→0.3）")
    ap.add_argument("--bench-select", action="store_true",
                    help="用 bench 10 帧选点：每 val 周期评 bench 全对帧数，存 bench_best.pt"
                         "（bench 帧不进训练梯度，仅作验收口径选点；报告需声明选点口径）")
    ap.add_argument("--human-select", action="store_true",
                    help="用 26 帧人工真值选点：每 val 周期分域评实物17/App9 错格，存 human_best.pt"
                         "（合计错格最少者入选；短程注入轮捕捉实物域高位+App 域爬升的交叉点）")
    ap.add_argument("--gen-select", action="store_true",
                    help="v33 泛化选点：每 val 周期评三轴泛化集(style/piece/angle 各300帧)"
                         "合计错格，存 gen_best.pt（泛化路线主选点口径；26帧/bench 仅监控不退）")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="计算精度：bf16=混合精度（v9 起实际默认，cuda 上 autocast bf16）；"
                         "fp32=全单精度（数值对照/排障用）")
    # v38a：格间注意力 + 占子辅助头（docs/v38性能研究与渲染提速方案-20261017.md §四）
    ap.add_argument("--attn", type=str2bool, default=False,
                    help="启用 90 token 格间注意力（ReZero 残差，主干权重命名不变可 init-from）")
    ap.add_argument("--xattn", type=str2bool, default=False,
                    help="v62 cross-attention 解码层（docs/v62方案；ReZero 零扰动起步）")
    ap.add_argument("--xattn-lr-mult", type=float, default=10.0,
                    help="xattn 新模块学习率倍率（v38a attn-lr-mult 先例）")
    ap.add_argument("--aux-head", type=str2bool, default=False,
                    help="启用占子辅助头（90 格 2 类，训练期损失约束）")
    ap.add_argument("--aux-weight", type=float, default=0.4, help="辅助头损失权重")
    ap.add_argument("--attn-lr-mult", type=float, default=10.0,
                    help="注意力分支学习率倍率（新初始化模块需高于主干微调水位）")
    # v41：格条件类平衡损失（侦察实锤：模型在每格内拟合多数类，(9,7)=R acc 仅 11.5%
    # 而 (9,8)=R 83.8%——格-类失衡的系统性修正）
    ap.add_argument("--cell-balance", type=str2bool, default=False,
                    help="启用格条件类平衡 CE（需 --cell-freq 频率表）")
    ap.add_argument("--cell-freq", default="",
                    help="格-类频率表 npz（stat_cell_freq.py 产出，[90,16] 每格归一化）")
    # v43 机制化早停（docs/v43实物贴片方案-20261017.md §三）：v42 实测损伤集中在最初 150 步、
    # 早期 checkpoint 可挽回 87% 退化——human26 合计错连续 N 个 val 周期无改善即存盘收工，
    # 即使数据再混入毒标签也自动止血。human_best.pt 每次刷新即存，早停不丢最优点。
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="human26 合计错连续 N 个 val 周期无改善则存盘早停（0=禁用；需 --human-select）")
    # v44 QAT 微调（docs/v44部署修复与实物扩量方案-20261017.md §1.3）：每步把 Conv/Linear
    # 权重 snap 到 int8 per-channel 对称网格再训练——模型在量化值域内重新优化，导出端
    # 静态量化时权重量化近无损、激活量化噪声已被容错。对照实验矩阵证实校准/格式层
    # 无双赢解（E3 bench+4 换 style-1.4），QAT 是部署口径（真机 int8）不伤四轴的正解。
    # 产出独立权重目录，现役轮零触碰；判据=int8 bench 与 int8 四轴，fp32 网格损失 ≤1pt 容忍。
    ap.add_argument("--qat", type=str2bool, default=False,
                    help="QAT 微调：每步前把权重 snap 到 int8 网格（配合低学习率与短程步数）")
    ap.add_argument("--qat-act", type=str2bool, default=False,
                    help="QAT-v2：叠加激活 fake-quant（STE，QUInt8 非对称 EMA range）——"
                         "v43q 权重 QAT 后 int8 bench 4 vs 网格 fp32 7，gap 全为激活噪声")
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(run_dir / "log.txt", "a", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    log(f"精度模式：{args.precision}（"
        + ("cuda autocast bf16" if args.precision == "bf16" and device == "cuda" else "fp32") + "）")

    train_fonts, gen_fonts = font_pools()
    log(f"训练字体 {len(train_fonts)} 个；泛化 held-out 字体 {len(gen_fonts)} 个: "
        f"{[Path(f).name for f in gen_fonts]}")
    assert len(train_fonts) >= 8 and len(gen_fonts) >= 2, "字体池不足（先跑 fetch_fonts.py）"

    # 加宽配置：v15 起支持更大容量；默认四元组 = v2~v14 全代际的 132K 配置
    width = tuple(int(x) for x in args.width.split(",")) if args.width else (64, 96, 160, 224)
    model = GridHead(width=width, head_dim=args.head_dim,
                     attn=args.attn, aux_head=args.aux_head, xattn=args.xattn).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数 {n_params/1e3:.1f}K"
        + (f"（attn={args.attn} aux={args.aux_head}）" if (args.attn or args.aux_head) else ""))

    ds = RenderDataset(args.seed, train_fonts, args.batch)
    dl = DataLoader(ds, batch_size=1, num_workers=args.workers,
                    prefetch_factor=3 if args.workers else None,
                    persistent_workers=args.workers > 0, pin_memory=True,
                    drop_last=False)  # dataset 已按批渲染，Loader 只做搬运

    # v3 真实帧混训数据（annotate_real.py 双域交叉标注；IGNORE=-100 的格训练时忽略）
    # v12 起 --real-npz 支持逗号分隔多集（annotate 标注集 + transplant 移植集）
    real_imgs_t = real_labs_t = None
    if args.real_npz and args.real_ratio > 0:
        real_npzs = [p for p in args.real_npz.split(",") if p and Path(p).exists()]
        if real_npzs:
            # v55 内存修复：逐池解压→转 NCHW 填入预分配缓冲→即弃。此前 listcomp+concatenate
            # 峰值 3× 池体积、NHWC 中间量 ri 存活到 main() 结束（稳态双份驻留）——12 池
            # 40k 帧配置 commit 20.3GB，watchdog 连爆的最大单项。labels 很小先扫一遍拿帧数。
            rl_parts, ns = [], []
            for p in real_npzs:
                with np.load(p, allow_pickle=False) as z:
                    rl_parts.append(z["labels"])
                    ns.append(z["labels"].shape[0])
            rl = np.concatenate(rl_parts)
            del rl_parts
            with np.load(real_npzs[0], allow_pickle=False) as z:
                _, _h, _w, _c = z["imgs"].shape
            buf = np.empty((sum(ns), _c, _h, _w), np.uint8)   # NCHW 唯一稳态驻留
            off = 0
            for p, n in zip(real_npzs, ns):
                with np.load(p, allow_pickle=False) as z:
                    arr = z["imgs"]                            # 单池 NHWC 瞬态，填入即弃
                buf[off:off + n] = arr.transpose(0, 3, 1, 2)
                off += n
            del arr
            real_imgs_t = torch.from_numpy(buf)
            real_labs_t = torch.from_numpy(rl.astype(np.int64))
            log(f"真实帧混训：{off} 样本（{len(real_npzs)} 集: {[Path(p).name for p in real_npzs]}），"
                f"每批占比 {args.real_ratio:.0%}")

    # v60 软标签蒸馏集：nano logits [N,90,16] fp16，与 real 池按序拼接对齐（帧序=imgs 帧序）
    soft_all_t = None
    if args.distill_soft:
        assert real_imgs_t is not None, "--distill-soft 需要 --real-npz 池在载（软标签挂在 real 帧上）"
        soft_npzs = [p for p in args.distill_soft.split(",") if p]
        real_npzs_ref = [p for p in args.real_npz.split(",") if p and Path(p).exists()]
        assert len(soft_npzs) == len(real_npzs_ref), \
            f"--distill-soft 清单({len(soft_npzs)})须与 --real-npz 池({len(real_npzs_ref)})按序一一对应"
        sparts, off_s = [], 0
        for p, rp in zip(soft_npzs, real_npzs_ref):
            with np.load(p, allow_pickle=False) as z:
                s = z["soft"]
            with np.load(rp, allow_pickle=False) as zr:
                nr = zr["labels"].shape[0]
            assert s.shape[0] == nr, \
                f"soft 帧 {s.shape[0]} ≠ 池帧 {nr}（{p} vs {rp}）——帧序错位，拒跑"
            sparts.append(s)
            off_s += s.shape[0]
        soft_all_t = torch.from_numpy(np.concatenate(sparts))       # [ΣN,90,16] fp16 驻留 ~126MB
        del sparts
        assert soft_all_t.shape[0] == real_imgs_t.shape[0], "soft 总帧数与 real 池不一致"
        log(f"软标签蒸馏：{off_s} 帧 nano logits 对齐 real 池"
            f"（T={args.distill_T} λ={args.distill_lambda}，KL 仅 valid 格）")

    # 类别权重：空格 0.8，占子 1.6（'x' 训练中不出现，权重无实际作用）
    w = torch.full((NCLASS,), 1.6, device=device)
    w[R.CLASS_INDEX["."]] = 0.8
    w[R.CLASS_INDEX["x"]] = 0.0

    # v38a 参数组：主干微调水位 / 注意力分支 ×mult（新初始化需快跑）/ 零衰减组
    # （ReZero α 初值 0 且 weight_decay 会把它往 0 拉，位置嵌入同理，aux 头无害但同组处理）
    # v62：xattn 新模块独立倍率组（xq/xk/xv/xproj/xffn）；α/β/kv_pos 零衰减+xattn 倍率
    named = dict(model.named_parameters())
    first = lambda n: n.split(".")[0]
    attn_first = ("wq", "wk", "wv", "proj", "ffn")
    xattn_first = ("xq", "xk", "xv", "xproj", "xffn")
    zerowd_a = ("a1", "a2", "pos_emb")                    # v38a 零衰减组
    zerowd_x = ("xa1", "xa2", "kv_pos_r", "kv_pos_c")     # v62 零衰减组
    zerowd_aux = ("aux_fc",)
    base_p = [p for n, p in named.items()
              if first(n) not in attn_first + xattn_first + zerowd_a + zerowd_x + zerowd_aux]
    groups = [{"params": base_p, "lr_mult": 1.0}]
    for names, mult, zerowd in ((attn_first, args.attn_lr_mult, False),
                                (xattn_first, args.xattn_lr_mult, False),
                                (zerowd_a, args.attn_lr_mult, True),
                                (zerowd_x, args.xattn_lr_mult, True),
                                (zerowd_aux, args.attn_lr_mult, True)):
        ps = [p for n, p in named.items() if first(n) in names]
        if ps:
            g = {"params": ps, "lr_mult": mult}
            if zerowd:
                g["weight_decay"] = 0.0
            groups.append(g)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    # v41 格条件类平衡权重表：w(cell, class) = clip(√(全局频率/格内频率), 1, 6)。
    # 格内频率→0（该格几乎不出此类）时用 cap 6 兜底；sqrt 温和化防少数类过矫正。
    # 注意：频率表必须与当前训练渲染分布一致（renderer 改动后须重跑 stat_cell_freq）。
    w_cell_class = None
    if args.cell_balance:
        assert args.cell_freq and Path(args.cell_freq).exists(), \
            "cell-balance 需要 --cell-freq 频率表（先跑 stat_cell_freq.py）"
        zf = np.load(args.cell_freq)
        freq = zf["freq"].astype(np.float64)                     # [90,16] 每格归一化
        glob_freq = freq.sum(axis=0) / max(1e-9, freq.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = glob_freq[None, :] / np.maximum(freq, 1e-6)
        ratio[~np.isfinite(ratio)] = 36.0                        # 0 频格-类 → cap²
        w_cell_class = torch.from_numpy(
            np.clip(np.sqrt(ratio), 1.0, 6.0)).float().to(device)
        log(f"格条件类平衡开启：{args.cell_freq}（{zf['n_samples']} 帧统计），"
            f"权重范围 [{w_cell_class.min():.2f}, {w_cell_class.max():.2f}]")

    # 验证集：同分布 + 泛化（held-out 字体）
    log("渲染验证集…")
    val_in = make_val_loader(args.seed, train_fonts, args.val_n, args.workers)
    val_gen = make_val_loader(args.seed + 1, gen_fonts, max(256, args.val_n // 2), args.workers)
    log("验证集就绪")

    # bench 选点集（验收口径；帧不进训练梯度）
    bench_imgs_t = bench_labs_t = None
    bench_best_score = (-1, 10 ** 9)   # (全对帧数, -总错格)
    if args.bench_select:
        bz = np.load(Path(__file__).parent / "data/eval_bench.npz", allow_pickle=False)
        bench_imgs_t = torch.from_numpy(np.ascontiguousarray(bz["imgs"].transpose(0, 3, 1, 2)))
        bench_labs_t = torch.from_numpy(bz["labels"].astype(np.int64))
        log(f"bench 选点开启：{len(bench_imgs_t)} 帧（每 val 周期评测，存 bench_best.pt）")

    # 26 帧人工真值选点（路边摊主口径；帧不进训练梯度）
    human_imgs_t = human_labs_t = None
    human_best_err = 10 ** 9
    human_stall = 0                  # v43 早停：human26 无改善的连续 val 周期数
    early_stop = False               # v49：early-stop 改 flag 移到 val 块尾 break——
                                     # 原位 break 会跳过末周期的 gen 巡检与 ckpt_val 存档
    deploy_rec: list[dict] = []      # v48 deploy_best 终局复选：每 val 周期一条记录
    if args.human_select:
        hz = np.load(Path(__file__).parent / "data/eval_photo_human.npz", allow_pickle=False)
        human_imgs_t = torch.from_numpy(np.ascontiguousarray(hz["imgs"].transpose(0, 3, 1, 2)))
        human_labs_t = torch.from_numpy(hz["labels"].astype(np.int64))
        log(f"human26 选点开启：{len(human_imgs_t)} 帧（每 val 周期分域评测，存 human_best.pt）")

    # v33 泛化三轴选点（style/piece/angle 各 300 帧；帧不进训练梯度）
    # v35 增第四轴 roboflow（留出贴片 300 帧）；轴文件缺失时跳过该轴，不崩训练
    gen_sel_imgs = gen_sel_labs = None
    gen_axis_names = gen_axis_lens = ()
    gen_best_err = 10 ** 9
    if args.gen_select:
        _gi, _gl, _names, _lens = [], [], [], []
        for _name in ("style", "piece", "angle", "roboflow"):
            _p = Path(__file__).parent / f"data/eval_gen_{_name}.npz"
            if not _p.exists():
                log(f"gen 轴 {_name} 缺失（{_p.name} 不存在），本次跳过")
                continue
            _z = np.load(_p, allow_pickle=False)
            _gi.append(_z["imgs"]); _gl.append(_z["labels"])
            _names.append(_name); _lens.append(len(_z["labels"]))
        gen_sel_imgs = torch.from_numpy(np.ascontiguousarray(np.concatenate(_gi).transpose(0, 3, 1, 2)))
        gen_sel_labs = torch.from_numpy(np.concatenate(_gl).astype(np.int64))
        gen_axis_names, gen_axis_lens = tuple(_names), tuple(_lens)
        log(f"gen 选点开启：{'/'.join(_names)} 共 {len(gen_sel_imgs)} 帧（每 val 周期评测，存 gen_best.pt）")

    step0, best_err0 = 0, -1.0
    ckpt_path, best_path = run_dir / "ckpt.pt", run_dir / "best.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step0, best_err0 = ck["step"], ck["best_err0"]
        log(f"续训：从 step {step0} 恢复（best full_err0={best_err0:.4f}）")

    steps_per_epoch = args.epoch_samples // args.batch
    total_steps = args.total_samples // args.batch

    # v3：从外部权重初始化（交集加载：只取当前模型中存在且形状一致的键，
    # 新增模块（attn/aux）保持初始化——ReZero α=0 保证加载后前向与主干完全等价）
    if args.init_from:
        init_ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = init_ck["model"]
        cur = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        n_skip = len(sd) - len(filtered)
        model.load_state_dict(filtered, strict=False)
        log(f"初始化：交集加载 {len(filtered)}/{len(cur)} 键自 {args.init_from}"
            f"（源 {len(sd)} 键，跳过 {n_skip} = 新增模块）")
    if args.qat:
        quantize_weights_(model)
        log("QAT 模式：初始权重已 snap 到 int8 网格——微调目标是网格上的最优解"
            "（fp32 口径的网格损失 ≤1pt 容忍，部署口径 int8 为准）")
    act_fq = None
    if args.qat and args.qat_act:
        act_fq = ActFakeQuant(model)
        log("QAT-v2：激活 fake-quant 已挂（Conv/Linear 输出 QUInt8 非对称 STE，EMA range）")

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    deadline = time.time() + args.max_minutes * 60 if args.max_minutes > 0 else None

    # v3：尺度增广下探。验证集已按 0.94~1.06（v2 口径）渲染完毕，此调整只影响训练流
    if args.scale_lo != 0.94:
        R.set_scale_range(args.scale_lo, 1.06)
        log(f"训练尺度增广改为 [{args.scale_lo}, 1.06]（验证集口径保持不变）")

    model.train()
    step = step0
    seen = step * args.batch
    t0 = time.time()
    loss_acc, acc_acc, n_acc = 0.0, 0.0, 0
    aux_acc, aux_n = 0.0, 0
    aux_on = args.aux_head  # 辅助头开关（构造于模型，损失路径据此分流）
    # 数据集为在线渲染的无限流（length=10^9），单层循环即可；时间盒/步数盒任一触发即停
    u8_iter = iter(dl)
    while True:
        if step >= total_steps or (deadline and time.time() > deadline):
            break
        if args.qat:
            quantize_weights_(model)   # QAT：每步前 snap 权重到 int8 网格
        if act_fq is not None and not act_fq.enabled and step >= 200:
            act_fq.enabled = True      # QAT-v2 热身完成（EMA range 已收敛），启用激活 fake-quant
        try:
            u8, lab = next(u8_iter)
        except StopIteration:  # 理论不可达（无限数据集），防御性重启迭代器
            u8_iter = iter(dl)
            continue
        u8, lab = u8[0], lab[0]  # dataset 按批渲染，Loader batch_size=1 → 解包
        B = u8.shape[0]
        # v3：每批前 k 行替换为真实样本（真实帧已含 IGNORE 标签，损失自动忽略）
        soft_b = None
        if real_imgs_t is not None:
            n_real = int(B * args.real_ratio)
            idx = torch.randint(0, real_imgs_t.shape[0], (n_real,))
            u8[:n_real] = real_imgs_t[idx]
            lab[:n_real] = real_labs_t[idx]
            if soft_all_t is not None:
                soft_b = soft_all_t[idx]          # v60：real 帧 nano logits 随批同行（CPU fp16）
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g.get("lr_mult", 1.0)
        u8 = u8.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        use_amp = device == "cuda" and args.precision == "bf16"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if aux_on:
                logits, aux = model(normalize_gpu(u8), return_aux=True)
            else:
                logits = model(normalize_gpu(u8))
                aux = None
            if w_cell_class is not None:
                # 手动加权 CE（含原 16 类权重 w；label_smoothing 在此路径弃用——
                # 格条件权重是主修正项，smoothing 与其叠加会稀释少数类梯度）
                logp = F.log_softmax(logits.reshape(-1, NCLASS), dim=1)
                lab_flat = lab.reshape(-1)
                valid_m = lab_flat != R.IGNORE_LABEL
                lab_safe = lab_flat.clamp_min(0)
                ll = logp.gather(1, lab_safe.unsqueeze(1)).squeeze(1)       # [B*90]
                cell_idx = torch.arange(ROWS * COLS, device=device).repeat(B)
                w_s = w[lab_safe] * w_cell_class[cell_idx, lab_safe]        # 类权重 × 格条件权重
                ce = -(ll * w_s)[valid_m].sum() / w_s[valid_m].sum()
            else:
                ce = F.cross_entropy(logits.reshape(-1, NCLASS), lab.reshape(-1),
                                     weight=w, ignore_index=R.IGNORE_LABEL,
                                     label_smoothing=0.05)
            loss = ce
            # v60 软标签蒸馏：real 帧 CE+λ·T²·KL（teacher=nano logits/T，仅 valid 格）
            # IGNORE 格跳过（teacher 对遮挡格无教导资格）；T² 补温度缩放的梯度折减
            if soft_all_t is not None and args.distill_lambda > 0:
                tl = soft_b.to(device).float() / args.distill_T          # [n_real,90,16]
                p_t = F.softmax(tl, dim=-1)
                q_t = F.log_softmax(logits[:n_real].float() / args.distill_T, dim=-1)
                kl_d = (p_t * (p_t.clamp_min(1e-9).log() - q_t)).sum(-1)  # [n_real,90]
                vmask = (lab[:n_real] != R.IGNORE_LABEL)
                kd = (kl_d * vmask).sum() / vmask.sum().clamp(min=1)
                loss = loss + args.distill_lambda * (args.distill_T ** 2) * kd
            # v38a 占子辅助损失：目标 = 该格是否占子（遮挡格 ignore），强迫特征对
            # "看得见"线性可分（漏检主导的直接约束）
            if aux is not None:
                tgt = (lab != R.CLASS_INDEX["."]).long()
                tgt = tgt.masked_fill(lab == R.IGNORE_LABEL, -100)
                aux_ce = F.cross_entropy(aux.reshape(-1, 2), tgt.reshape(-1),
                                         ignore_index=R.IGNORE_LABEL)
                loss = loss + args.aux_weight * aux_ce
                aux_acc += float(aux_ce.detach()); aux_n += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_acc += float(loss.detach()); n_acc += 1
        with torch.no_grad():
            valid = lab != R.IGNORE_LABEL
            acc_acc += float(((logits.argmax(2) == lab) & valid).sum() / valid.sum().clamp(min=1))
        step += 1
        seen += args.batch

        if step % args.log_every == 0:
            rate = seen / max(1e-6, time.time() - t0)
            aux_str = f" aux={aux_acc/aux_n:.4f}" if aux_n else ""
            log(f"step {step}/{total_steps} loss={loss_acc/n_acc:.4f} "
                f"cell_acc={acc_acc/n_acc:.4f} lr={lr_at(step):.2e} "
                f"rate={rate:.0f}样本/s{aux_str}")
            loss_acc = acc_acc = 0.0; n_acc = 0
            aux_acc, aux_n = 0.0, 0

        if step % args.val_every == 0:
            rec = {"step": step}     # v48 deploy 复选记录：三选点线各段填充
            m_in = evaluate(model, val_in, device)
            m_gen = evaluate(model, val_gen, device)
            log(f"[val] in-distribution: cell={m_in['cell_acc']:.4f} occ={m_in['occ_acc']:.4f} "
                f"err0={m_in['full_err0']:.3f} err1={m_in['full_err1']:.3f} | "
                f"heldout-fonts: cell={m_gen['cell_acc']:.4f} occ={m_gen['occ_acc']:.4f} "
                f"err0={m_gen['full_err0']:.3f}")
            if m_in["full_err0"] > best_err0:
                best_err0 = m_in["full_err0"]
                torch.save({"model": model.state_dict(), "step": step,
                            "metrics": {"in": m_in, "gen": m_gen}}, best_path)
                log(f"[val] 新最优 full_err0={best_err0:.4f} → best.pt")

            if bench_imgs_t is not None:
                model.eval()
                with torch.no_grad():
                    blogits = model(normalize_gpu(bench_imgs_t.to(device)))
                    bpred = blogits.argmax(dim=2)
                    bvalid = bench_labs_t.to(device) != R.IGNORE_LABEL
                    berr = ((bpred != bench_labs_t.to(device)) & bvalid).sum(dim=1)
                model.train()
                score = (int((berr == 0).sum()), -int(berr.sum()))
                rec["bench"] = {"full": score[0], "err": -score[1]}
                # v48 无条件巡检日志：选点盲区实锤（v47 step150→200 实物 +5 格全程无日志）
                log(f"[bench巡检] step {step}: 全对 {score[0]}/10，总错格 {-score[1]}"
                    + ("（刷新）" if score > bench_best_score else ""))
                if score > bench_best_score:
                    bench_best_score = score
                    torch.save({"model": model.state_dict(), "step": step,
                                "bench": {"errs": berr.tolist(), "score": score}}, run_dir / "bench_best.pt")
                    log(f"[bench选点] 全对 {score[0]}/10，总错格 {-score[1]} → bench_best.pt "
                        f"(step {step})")
                    if score[0] == len(berr):
                        # v51：满分不再立即停止（v50 实证——高起点 init-from 语境下该
                        # 截断杀掉 human26 改善期：v49 甜点 900 步 vs v50 被 350 打断、
                        # human_best 冻结 step50）。bench_best.pt 已固化，继续训练，
                        # 停止唯一由 early-stop-patience 决定。
                        log("[bench选点] ★ bench 10/10 全对——验收达成，继续训练"
                            "（v51 起满分不停止，early-stop 唯一裁决）")

            if human_imgs_t is not None:
                model.eval()
                with torch.no_grad():
                    hlogits = model(normalize_gpu(human_imgs_t.to(device)))
                    hpred = hlogits.argmax(dim=2)
                    hlab = human_labs_t.to(device)
                    hvalid = hlab != R.IGNORE_LABEL
                    hocc = hvalid & (hlab != R.CLASS_INDEX["."])
                    herr = (hpred != hlab) & hvalid          # [26,90]
                model.train()
                # 分域：0~16 实物照片 / 17~25 App·教学图（与 .cache/human26_eval.py 口径一致）
                r_err = int(herr[:17].sum()); r_occ = int(herr[:17][hocc[:17]].sum())
                a_err = int(herr[17:].sum()); a_occ = int(herr[17:][hocc[17:]].sum())
                total = int(herr.sum())
                rec["human_total"] = total
                # v48 无条件巡检日志（选点盲区实锤）：每周期记录真实错数，非刷新周期不再丢失
                log(f"[human巡检] step {step}: 合计 {total} 错（实物17 cell {r_err}/occ {r_occ}，"
                    f"App9 cell {a_err}/occ {a_occ}）")
                if total < human_best_err:
                    human_best_err = total
                    human_stall = 0
                    torch.save({"model": model.state_dict(), "step": step,
                                "human": {"real17": {"cell_err": r_err, "occ_err": r_occ},
                                          "app9": {"cell_err": a_err, "occ_err": a_occ},
                                          "total_err": total}}, run_dir / "human_best.pt")
                    log(f"[human选点] 合计 {total} 错（实物17 cell {r_err}/occ {r_occ}，"
                        f"App9 cell {a_err}/occ {a_occ}）→ human_best.pt (step {step})")
                elif args.early_stop_patience > 0:
                    human_stall += 1
                    if human_stall >= args.early_stop_patience:
                        log(f"[early-stop] human26 合计错已 {human_stall} 个 val 周期无改善"
                            f"（best {human_best_err}）——机制化早停收工 (step {step})；"
                            f"最优点已存 human_best.pt")
                        early_stop = True   # v49：break 移到 val 块尾（补全末周期存档）

            if gen_sel_imgs is not None:
                model.eval()
                with torch.no_grad():
                    _gerr = []
                    for _s in range(0, len(gen_sel_imgs), 96):
                        _lo = model(normalize_gpu(gen_sel_imgs[_s:_s + 96].to(device)))
                        _pd = _lo.argmax(dim=2)
                        _lb = gen_sel_labs[_s:_s + 96].to(device)
                        _vd = _lb != R.IGNORE_LABEL
                        _gerr.append((_pd != _lb) & _vd)
                    gerr = torch.cat(_gerr)                      # [Σ各轴帧,90]
                model.train()
                # 按加载时记录的各轴帧数切分（加轴无需改此处）
                _off, _axes = 0, {}
                for _n, _ln in zip(gen_axis_names, gen_axis_lens):
                    _axes[_n] = int(gerr[_off:_off + _ln].sum())
                    _off += _ln
                g_total = int(gerr.sum())
                # v48 deploy 复选口径：train 内每轴 occ（公式同 eval_gen，帧数为选点子集，
                # 仅用于周期间相对排序与容差判断，终值以训后三把尺为准）
                _gocc, _goff = [], 0
                for _n, _ln in zip(gen_axis_names, gen_axis_lens):
                    _lb = gen_sel_labs[_goff:_goff + _ln].to(device)
                    _vd = _lb != R.IGNORE_LABEL
                    _oc = _vd & (_lb != R.CLASS_INDEX["."])
                    _gocc.append(1.0 - float((gerr[_goff:_goff + _ln] & _oc).sum())
                                 / max(1, int(_oc.sum())))
                    _goff += _ln
                rec["gen_avg"] = sum(_gocc) / len(_gocc)
                rec["gen_total"] = g_total   # v49 复选 tiebreak 口径：cell 全错格（含幻视）
                # v48 无条件巡检日志（选点盲区实锤）：非刷新周期的 gen 错数不再丢失
                log(f"[gen巡检] step {step}: {len(_axes)} 轴 {g_total} 错（"
                    + "/".join(f"{k} {_axes[k]}" for k in gen_axis_names)
                    + f"）avg_occ={rec['gen_avg']:.4f}")
                if g_total < gen_best_err:
                    gen_best_err = g_total
                    torch.save({"model": model.state_dict(), "step": step,
                                "gen": {**{f"{k}_err": v for k, v in _axes.items()},
                                        "total_err": g_total}},
                               run_dir / "gen_best.pt")
                    log(f"[gen选点] {len(_axes)} 轴 {g_total} 错（"
                        + "/".join(f"{k} {_axes[k]}" for k in gen_axis_names)
                        + f"）→ gen_best.pt (step {step})")

            # 每 val 周期落 ckpt（含优化器状态）：watchdog 保命终止（commit 超 58GB kill_tree）
            # 属强杀、主循环尾部的正常保存不会执行——没有这里，被终止轮只能从零重训
            #（v39 首点火实测 15 分钟白跑）。选点 best.pt 系列本就周期保存，此处补 ckpt。
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "best_err0": best_err0}, ckpt_path)
            # v48 deploy 复选配套：每 val 周期存 model-only 快照（~11MB），终局复选后清理
            deploy_rec.append(rec)
            torch.save({"model": model.state_dict(), "step": step},
                       run_dir / f"ckpt_val{step}.pt")
            if early_stop:   # v49：早停统一在块尾 break（末周期 gen 巡检+快照已补全）
                break

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "best_err0": best_err0}, ckpt_path)
    # v49 复选重写（v48 实战缺陷清算，红队+复选修正卡双实锤）：gen_avg 只算占用格
    # 盲幻视（step 350 在 occ 排序第 1、gen 总错排序垫底）；容差预过滤把候选集收缩
    # 成单点致主排序失效。改为 human26 主排序优先 + gen 总错 tiebreak + 崩盘兜底。
    # v56 复选键再升级（v55 跷跷板实证）：字典序在"26 帧↔四轴"跷跷板轮退化为
    # human26 单口径最优（v55 选 step 450，而训练后段 850-1150 四轴好 289 错、
    # human26 仅 +2 格——双优平衡点被跳过）。改两把尺 rank 融合（rank 和最小，
    # tie 归 human26）：权重插值实验（soup05/07）证明双优点存在且 rank 融合可选中
    # 它。崩盘兜底保留（rank 融合天然抑制，防线只在单侧崩坏时触发）。
    if deploy_rec:
        _cand = [r for r in deploy_rec if r.get("human_total") is not None
                 and r.get("gen_total") is not None]
        if _cand:
            _hmin = min(r["human_total"] for r in _cand)
            _gmin = min(r["gen_total"] for r in _cand)
            _by_h = sorted(_cand, key=lambda r: (r["human_total"], r["gen_total"], r["step"]))
            _by_g = sorted(_cand, key=lambda r: (r["gen_total"], r["human_total"], r["step"]))
            _hrank = {id(r): i for i, r in enumerate(_by_h)}
            _grank = {id(r): i for i, r in enumerate(_by_g)}
            _dep = min(_cand, key=lambda r: (_hrank[id(r)] + _grank[id(r)],
                                             r["human_total"], r["gen_total"]))
            if _dep["gen_total"] > _gmin + 400:          # human26 优侧 gen 崩盘兜底
                _dep = min(_cand, key=lambda r: (r["gen_total"], r["human_total"]))
            _src = run_dir / f"ckpt_val{_dep['step']}.pt"
            if _src.exists():
                shutil.copy(_src, run_dir / "deploy_best.pt")
                log(f"[deploy复选] {len(deploy_rec)} 周期全局复选（rank 融合）→ step {_dep['step']}"
                    f"（human26 {_dep['human_total']} 错，gen四轴总错 {_dep['gen_total']}，"
                    f"bench {_dep.get('bench', {}).get('full', '—')}/10）"
                    f"→ deploy_best.pt (step {_dep['step']})")
    for r in deploy_rec:   # 临时快照清理：无论复选是否成功都清（防 _cand 空时残留累积）
        (run_dir / f"ckpt_val{r['step']}.pt").unlink(missing_ok=True)
    log(f"停止于 step {step}（{seen} 样本），best full_err0={best_err0:.4f}；ckpt/best 已保存")
    log_f.close()


if __name__ == "__main__":
    main()
