# -*- coding: utf-8 -*-
"""训练全流程一键编排器（省 token 基建）：预检 → watchdog → 训练 → 训后评测导出链 → 自动报告。

用法（cwd 任意，pipeline 内部统一以本目录为工作目录）:
  python pipeline.py --preset v35 --run-name v36
  python pipeline.py --preset v36 --run-name v36 --baseline runs/v35/auto_summary.json
  python pipeline.py --preset v35 --run-name v35replay --override "total-samples=4800,workers=4"

流程:
  阶段0 预检   real-npz 逐个存在性（根治 train.py 静默跳过坑）/评测集/磁盘/commit 基线/
               net2net 初始化自动生成；任一硬性缺失立即退出不点火
  阶段1 watchdog 守护线程  每 120s 查系统 commit（GlobalMemoryStatusEx，与 v35_watchdog.sh
               的 Win32_OperatingSystem 口径等价），写 runs/<run>/watchdog.log；
               > --commit-warn(55GB) 记警报，> --commit-kill(58GB) 自动终止训练保命
               （ckpt.pt 每 val 周期已存盘，重跑同命令自动续训；--no-guard 关闭）
  阶段2 训练   子进程跑 train.py（preset.train 段 + --override 覆盖）；bench 10/10 提前
               结束按成功处理；中断/断电后重跑同一命令即从 ckpt.pt 续训
  阶段3 训后链 export fp32 三连（gen/bench/human_best，导完立即改名——根治固定名互相
               覆盖坑）→ gen_best 基座 int8（--calib-real）→ eval_gen 四轴（fp32 三选点
               +int8+nano）→ eval_human26 → eval_bench；单步失败记档继续，不互相拖死
  阶段4 汇总   runs/<run>/auto_summary.json（机器可读，供下轮 --baseline 对拍）+
               自动报告.md（四轴/26帧/bench/int8/判据 PASS-FAIL/与上轮 delta）+ DONE/FAILED 标记

训练数字从 runs/<run>/log.txt 正则抽取（选点 step 取该文件名最后一次出现的 step，与 .pt
覆盖写语义一致）。评测报告 json 落 runs/<run>/ 内，data/ 只留历史手工报告。
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
NANO = HERE / "../../app/src/main/assets/models/nano_v3-0319.onnx"
CLS_W, CLS_H = 280, 315  # 与 renderer 口径一致，仅用于报告头

# ---------------- 基础工具 ----------------

class RunLog:
    """pipeline 自身日志：stdout + runs/<run>/pipeline.log 双写。"""

    def __init__(self, run_dir: Path):
        self.f = open(run_dir / "pipeline.log", "a", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"[pipeline {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def commit_gb() -> float:
    """系统 commit charge（GB）：GlobalMemoryStatusEx 的 TotalPageFile-AvailPageFile，
    与 v35_watchdog.sh 用的 Win32_OperatingSystem Total/FreeVirtualMemory 同口径
    （物理内存+页面文件，v20 蓝屏教训的监控指标）。非 Windows 返回 -1。"""
    if os.name != "nt":
        return -1.0

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return -1.0
    return (st.ullTotalPageFile - st.ullAvailPageFile) / 2**30


def deep_get(d: dict, path: str):
    """点路径取值：auto_summary 的 metric 路径（如 gen.style.occ）。取不到返回 None。"""
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# ---------------- 阶段 0：预检 ----------------

def precheck(preset: dict, args, log: RunLog) -> list[str]:
    """返回 train.py 参数列表；硬性缺失直接 raise SystemExit。"""
    run_dir = HERE / "runs" / args.run_name
    tr = preset.get("train", {})
    fail: list[str] = []

    # 1) real-npz 逐个存在性（train.py 对缺失池静默跳过，必须在点火前拦住）
    for p in tr.get("real-npz", []):
        if not (HERE / p).exists():
            fail.append(f"真实帧池缺失: {p}")

    # 2) 选点评测集存在性（选点开着但评测集缺 → 选点形同虚设）
    selects = tr.get("select", [])
    if "bench-select" in selects and not (HERE / "data/eval_bench.npz").exists():
        fail.append("data/eval_bench.npz 缺失（bench-select 开启中）")
    if "human-select" in selects and not (HERE / "data/eval_photo_human.npz").exists():
        fail.append("data/eval_photo_human.npz 缺失（human-select 开启中）")
    if "gen-select" in selects:
        axes = [n for n in ("style", "piece", "angle", "roboflow")
                if (HERE / f"data/eval_gen_{n}.npz").exists()]
        if not axes:
            fail.append("data/eval_gen_*.npz 全部缺失（gen-select 开启中）")
        else:
            log(f"gen 选点轴就绪: {'/'.join(axes)}")

    # 3) net2net 初始化自动生成（不存在才生成；断点续训/复跑直接复用）
    n2n = preset.get("net2net", {})
    init_pt = run_dir / "net2net_init.pt"
    if n2n.get("src") and not args.no_n2n:
        if init_pt.exists():
            log(f"net2net 初始化已存在，复用 {init_pt.name}")
        elif (HERE / n2n["src"]).exists():
            log(f"生成 net2net 初始化: {n2n['src']} → {init_pt.relative_to(HERE)}")
            cmd = [PY, str(HERE / "net2net_widen.py"), "--src", n2n["src"],
                   "--out", str(init_pt),
                   "--width", tr.get("width", ""), "--head-dim", str(tr.get("head-dim", 64))]
            r = subprocess.run([c for c in cmd if c != ""], cwd=HERE,
                               capture_output=True, text=True)
            if r.returncode != 0:
                fail.append(f"net2net_widen 失败: {r.stdout[-400:]} {r.stderr[-400:]}")
        else:
            fail.append(f"net2net 源 ckpt 缺失: {n2n['src']}")

    # 4) 磁盘与 commit 基线
    du = shutil.disk_usage(HERE)
    if du.free < 20 * 2**30:
        fail.append(f"磁盘剩余 {du.free/2**30:.0f}GB < 20GB")
    gb = commit_gb()
    if gb >= 0:
        log(f"commit 基线 {gb:.1f}GB（红线 warn {args.commit_warn}/kill {args.commit_kill}）")
        if gb > args.commit_warn:
            fail.append(f"点火前 commit 已 {gb:.1f}GB > {args.commit_warn}GB，先清内存")

    # 5) nano 对照模型（评测链要用，缺失只降级不拦截）
    if not NANO.exists():
        log(f"警告: nano 对照模型不存在（{NANO}），评测链将跳过 nano 对比")

    if fail:
        for f_ in fail:
            log(f"预检失败: {f_}")
        raise SystemExit("预检未通过，拒绝点火（详见 pipeline.log）")

    # 6) 组装 train.py 参数
    cmd = [PY, str(HERE / "train.py"), "--run-name", args.run_name]
    for k, v in tr.items():
        if k == "select":
            cmd += [f"--{s}" for s in v]
        elif k == "real-npz":
            cmd += ["--real-npz", ",".join(v)]
        else:
            cmd += [f"--{k}", str(v)]
    if init_pt.exists() and n2n.get("src") and not args.no_n2n:
        cmd += ["--init-from", str(init_pt.relative_to(HERE))]
    if args.precision:
        cmd += ["--precision", args.precision]
    for kv in filter(None, args.override.split(",")):
        k, _, v = kv.partition("=")
        k = k.strip().lstrip("-")
        cmd = [c for c in cmd if c not in (f"--{k}",)]
        cmd += [f"--{k}", v]
        log(f"覆盖参数: --{k} {v}")
    if args.extra:
        cmd += args.extra.split()
    return cmd


# ---------------- 阶段 1：watchdog ----------------

def watchdog(run_dir: Path, train_proc: subprocess.Popen, args, log: RunLog,
             stop: threading.Event):
    """每 120s 查 commit 写 watchdog.log；超 kill 线终止训练子进程。"""
    wf = open(run_dir / "watchdog.log", "a", encoding="utf-8")
    peak = 0.0
    while not stop.is_set():
        gb = commit_gb()
        if gb < 0:   # 非 Windows：无监控口径，空转等待即可
            stop.wait(120)
            continue
        peak = max(peak, gb)
        mark = "OK"
        if gb > args.commit_kill:
            mark = f"!! {gb:.1f}GB 超 kill 线 {args.commit_kill}GB，自动终止训练保命"
        elif gb > args.commit_warn:
            mark = f"! {gb:.1f}GB 超 warn 线 {args.commit_warn}GB"
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] commit={gb:.1f}GB peak={peak:.1f}GB {mark}"
        print(f"[watchdog] {line}", flush=True)
        wf.write(line + "\n")
        wf.flush()
        if gb > args.commit_kill and not args.no_guard:
            try:
                train_proc.terminate()
                log(f"watchdog 已终止训练子进程（commit {gb:.1f}GB > {args.commit_kill}GB）")
            except OSError:
                pass
            break
        stop.wait(120)
    wf.close()


# ---------------- 阶段 2：训练 ----------------

def run_train(cmd: list, run_dir: Path, args, log: RunLog) -> int:
    log("训练点火: " + " ".join(cmd))
    log(f"进度查看: tail {run_dir / 'log.txt'}（或问我一次即可）")
    proc = subprocess.Popen(cmd, cwd=HERE)
    stop = threading.Event()
    wd = threading.Thread(target=watchdog,
                          args=(run_dir, proc, args, log, stop), daemon=True)
    wd.start()
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        log("收到中断，终止训练子进程…")
        proc.terminate()
        rc = proc.wait()
    stop.set()
    wd.join(timeout=5)
    log(f"训练子进程退出 rc={rc}")
    return rc


# ---------------- 阶段 3：训后链 ----------------

def _sh(cmd: list, log: RunLog, tag: str, out_file: Path | None = None) -> bool:
    """跑一条子命令；stdout/stderr 收进 runs/<run>/post_<tag>.log；失败记档不抛。"""
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    blob = (r.stdout or "") + (("\n--- stderr ---\n" + r.stderr) if r.stderr else "")
    if out_file:
        out_file.write_text(blob, encoding="utf-8")
    if r.returncode != 0:
        log(f"[{tag}] 失败 rc={r.returncode}（详见 post_{tag}.log）: {blob[-500:]}")
        return False
    log(f"[{tag}] 完成")
    return True


def post_chain(preset: dict, args, log: RunLog) -> dict:
    """返回 {export: {...}} 摘要；单步失败降级继续。"""
    run_dir = HERE / "runs" / args.run_name
    rel = lambda p: str(Path(p).relative_to(HERE))
    tr = preset.get("train", {})
    width, head_dim = tr.get("width", ""), str(tr.get("head-dim", 64))
    exp_cfg = preset.get("export", {})
    # v48：deploy_best（终局复选快照，train.py 产出）加入导出与三把尺口径；
    # 缺失（watchdog 强杀时复选未跑）自动跳过，安全降级
    sel_pts = [n for n in ("gen", "human", "bench", "deploy")
               if (run_dir / f"{n}_best.pt").exists()]
    # v38a：attn/aux 开关透传给 export（模型构造须与训练一致，否则 state_dict 不匹配）
    attn_flags = []
    if str(tr.get("attn", "")).strip().lower() in ("1", "true", "yes"):
        attn_flags += ["--attn", "True"]
    if str(tr.get("aux-head", "")).strip().lower() in ("1", "true", "yes"):
        attn_flags += ["--aux-head", "True"]
    if str(tr.get("xattn", "")).strip().lower() in ("1", "true", "yes"):
        attn_flags += ["--xattn", "True"]   # v62：xattn 开关透传导出（漏传=state_dict 崩）
    summary: dict = {}

    # 3a. fp32 三连导出（每次导完立即改名，根治 gridhead_fp32.onnx 互相覆盖坑）
    for name in sel_pts:
        tag = f"export_fp32_{name}"
        ok = _sh([PY, str(HERE / "export.py"), "--ckpt", rel(run_dir / f"{name}_best.pt"),
                  "--width", width, "--head-dim", head_dim, "--skip-quant", *attn_flags],
                 log, tag, run_dir / f"post_{tag}.log")
        src, dst = run_dir / "gridhead_fp32.onnx", run_dir / f"{name}_fp32.onnx"
        if ok and src.exists():
            shutil.move(src, dst)
            rep = run_dir / "export_report.json"
            if rep.exists():  # 报告随模型一起改名归档，避免三连互相覆盖
                shutil.move(rep, run_dir / f"{name}_fp32_export_report.json")
            log(f"[{tag}] → {dst.name}")
        else:
            log(f"[{tag}] 未产出 onnx，跳过改名")

    # 3b. int8（v45 口径：双基座——gen_best 四轴口径 + human_best 部署口径；
    # v43/v44 两轮实测 human_best 是部署甜点（26 帧分叉轮 gen_best 43 vs human_best 36），
    # 部署口径统一 human_best，gates 的 bench_int8 引 int8_hb 行；export_report 随模型改名）
    int8_jobs = []
    # human_best 必须先导出改名 _hb，gen_best 后导出保留原名——反过来 job 2 会覆盖
    # gridhead_int8.onnx（v45 实证：gen_best 基座 int8 产物丢失、两键同源）
    if (run_dir / "human_best.pt").exists():
        int8_jobs.append(("human_best", "export_int8_hb", "gridhead_int8_hb.onnx",
                          "human_best_int8_export_report.json"))
    if (run_dir / "gen_best.pt").exists():
        int8_jobs.append(("gen_best", "export_int8", None, None))
    for ck_name, tag, rename_onnx, rename_rep in int8_jobs:
        cmd = [PY, str(HERE / "export.py"), "--ckpt", rel(run_dir / f"{ck_name}.pt"),
               "--width", width, "--head-dim", head_dim, *attn_flags]
        if exp_cfg.get("calib-real") is not None:
            cmd += ["--calib-real", str(exp_cfg["calib-real"])]
        ok = _sh(cmd, log, tag, run_dir / f"post_{tag}.log")
        if ok and rename_onnx:
            src, dst = run_dir / "gridhead_int8.onnx", run_dir / rename_onnx
            if src.exists():
                shutil.move(src, dst)
                rep = run_dir / "export_report.json"
                if rep.exists():  # 报告随模型改名归档，防两次导出互相覆盖
                    shutil.move(rep, run_dir / rename_rep)
        else:
            log(f"[{tag}] 未产出，跳过改名")

    # 3c~3e. 三把尺全量对拍（fp32 三选点 + int8 双基座 + nano）
    models = [run_dir / f"{n}_fp32.onnx" for n in sel_pts]
    if (run_dir / "gridhead_int8.onnx").exists():
        models.append(run_dir / "gridhead_int8.onnx")
    if (run_dir / "gridhead_int8_hb.onnx").exists():
        models.append(run_dir / "gridhead_int8_hb.onnx")
    models = [rel(m) for m in models if m.exists()]
    if NANO.exists():
        models.append(rel(NANO))
    if models:
        _sh([PY, str(HERE / "eval_gen.py"), *models, "--json", rel(run_dir / "eval_gen.json")],
            log, "eval_gen", run_dir / "post_eval_gen.log")
        _sh([PY, str(HERE / "eval_human26.py"), *models, "--json", rel(run_dir / "eval_human26.json")],
            log, "eval_human26", run_dir / "post_eval_human26.log")
        # v50 评测双轨：v2 干净版（19 格标注修正）只作赛后裁判，train.py 选点仍挂 v1
        _sh([PY, str(HERE / "eval_human26.py"), *models, "--npz", "data/eval_photo_human_v2.npz",
             "--json", rel(run_dir / "eval_human26_v2.json")],
            log, "eval_human26_v2", run_dir / "post_eval_human26_v2.log")
        # v52 评测三口径：gold 68 帧（26 v2 帧+42 帧双模型双高共识帧）
        _sh([PY, str(HERE / "eval_human26.py"), *models, "--npz", "data/eval_human_consensus49.npz",
             "--json", rel(run_dir / "eval_human_gold49.json")],
            log, "eval_human_gold68", run_dir / "post_eval_human_gold49.log")
        _sh([PY, str(HERE / "eval_bench.py"), *models, "--json", rel(run_dir / "eval_bench.json")],
            log, "eval_bench", run_dir / "post_eval_bench.log")
    else:
        log("训后链：无可用 onnx（训练可能失败），评测全跳过")
    summary["selects_exported"] = sel_pts
    return summary


# ---------------- 阶段 4：汇总 ----------------

def parse_train_log(run_dir: Path) -> dict:
    """从 log.txt 抽选点 step（该文件名最后一次出现处，与 .pt 覆盖写语义一致）与最终行。"""
    out = {"gen_step": None, "human_step": None, "bench_step": None,
           "deploy_step": None, "final_step": None, "best_err0": None}
    p = run_dir / "log.txt"
    if not p.exists():
        return out
    txt = p.read_text(encoding="utf-8", errors="replace")
    for key, name in (("gen_step", "gen_best"), ("human_step", "human_best"),
                      ("bench_step", "bench_best"), ("deploy_step", "deploy_best")):
        hits = re.findall(rf"→ {name}\.pt \(step (\d+)\)", txt)
        if hits:
            out[key] = int(hits[-1])
    m = re.findall(r"停止于 step (\d+)（\d+ 样本），best full_err0=([\d.]+)", txt)
    if m:
        out["final_step"], out["best_err0"] = int(m[-1][0]), float(m[-1][1])
    early = "★ bench 10/10 全对" in txt
    out["early_stop_bench_full"] = early
    return out


def pick_gen(report: dict, key_prefix: str) -> dict | None:
    """eval_gen.json/eval_bench.json 里按文件名前缀挑模型行。
    键形如 'v36/gen_fp32.onnx'（目录段各家不同），故只按文件名部分匹配。"""
    fname = Path(key_prefix).name
    # 精确匹配优先：'gridhead_int8' 是 'gridhead_int8_hb' 的前缀，纯 startswith
    # 会让 bench_int8 键静默吃到 _hb 行（v45 实证）
    for k, v in report.items():
        if Path(k).name == fname:
            return v
    for k, v in report.items():
        if Path(k).name.startswith(fname):
            return v
    return None


def pick_human26(report: dict, key_prefix: str) -> dict | None:
    """eval_human26.json 挑模型行并把中文域键映射为 real17/app9/total。"""
    fname = Path(key_prefix).name
    for k, v in report.items():
        if Path(k).name.startswith(fname):
            return {"real17": v.get("实物17"), "app9": v.get("App域9"),
                    "total": v.get("合计26")}
    return None


def summarize(preset: dict, args, train_rc: int, log: RunLog) -> bool:
    run_dir = HERE / "runs" / args.run_name
    name = lambda prefix: f"{args.run_name}/{prefix}"
    summary: dict = {"run": args.run_name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "train_rc": train_rc, "precision": args.precision or "bf16"}

    tl = parse_train_log(run_dir)
    summary["train"] = tl
    summary["select"] = {k: tl[f"{k}_step"] for k in ("gen", "human", "bench", "deploy")}
    # v48 健康监控代码化：选点过早红旗（v47 实证选点分叉+过程不可见=退步通道，
    # 此前是纯人工纪律零代码；选点 step≤100 意味着 warmup 段即锁定，须人工复核）
    if tl.get("human_step") is not None and tl["human_step"] <= 100:
        summary["health"] = {"redflag": f"human 选点 step={tl['human_step']}≤100"
                                        "——选点过早，须人工核对 deploy 复选与实物域走向"}
        log(f"[健康红旗] {summary['health']['redflag']}")

    # 四轴 gen（gen_best fp32 / int8 / nano）
    gpath = run_dir / "eval_gen.json"
    if gpath.exists():
        g = json.loads(gpath.read_text(encoding="utf-8"))
        for key, prefix in (("gen", f"{args.run_name}/gen_fp32"),
                            ("gen_human_fp32", f"{args.run_name}/human_fp32"),
                            ("gen_bench_fp32", f"{args.run_name}/bench_fp32"),
                            ("gen_deploy", f"{args.run_name}/deploy_fp32"),
                            ("gen_int8", "gridhead_int8"), ("gen_nano", "nano_v3")):
            row = pick_gen(g, prefix)
            if row:
                summary[key] = {a: row[a] for a in ("style", "piece", "angle", "roboflow")
                                if a in row}

    # 26 帧 / bench（gen_best 主口径 + human_best/bench_best 选点口径，其余模型留原始 json）
    hpath = run_dir / "eval_human26.json"
    if hpath.exists():
        h = json.loads(hpath.read_text(encoding="utf-8"))
        row = pick_human26(h, f"{args.run_name}/gen_fp32")
        if row:
            summary["human26"] = row
        row = pick_human26(h, "human_fp32")
        if row:
            summary["human26_human_best"] = row
        # v50 评测双轨：v2 干净版（19 格标注修正）主判据口径
        h2path = run_dir / "eval_human26_v2.json"
        if h2path.exists():
            h2 = json.loads(h2path.read_text(encoding="utf-8"))
            row = pick_human26(h2, f"{args.run_name}/gen_fp32")
            if row:
                summary["human26v2"] = row
            row = pick_human26(h2, "human_fp32")
            if row:
                summary["human26v2_human_best"] = row
            row = pick_human26(h2, f"{args.run_name}/deploy_fp32")
            if row:
                summary["human26v2_deploy"] = row
        # v52：gold 68 帧共识口径（分域=前 26 帧 v2 域/后 42 帧共识帧，total 为一致性主指标）
        gpath = run_dir / "eval_human_gold49.json"
        if gpath.exists():
            g68 = json.loads(gpath.read_text(encoding="utf-8"))
            row = pick_human26(g68, f"{args.run_name}/human_fp32")
            if row:
                summary["gold49_human_best"] = row
            row = pick_human26(g68, "nano_v3")
            if row:
                summary["gold49_nano"] = row
        # v48：deploy_best 复合快照口径（终局复选=部署基座）
        row = pick_human26(h, f"{args.run_name}/deploy_fp32")
        if row:
            summary["human26_deploy"] = row
    bpath = run_dir / "eval_bench.json"
    if bpath.exists():
        b = json.loads(bpath.read_text(encoding="utf-8"))
        row = pick_gen(b, f"{args.run_name}/gen_fp32")
        if row:
            summary["bench"] = {"full": row["full"], "total_err": row["total_err"]}
        row = pick_gen(b, "bench_fp32")
        if row:
            summary["bench_bench_best"] = {"full": row["full"], "total_err": row["total_err"]}
        # v48：deploy_best 复合快照的 bench 口径
        row = pick_gen(b, f"{args.run_name}/deploy_fp32")
        if row:
            summary["bench_deploy"] = {"full": row["full"], "total_err": row["total_err"]}
        # v44：int8 bench 上报告+可设门——部署口径（真机跑的就是 int8）此前九轮
        # 被掩盖（v35 0/10 … v43 2/10，fp32 5~8），本行拆掉盲区（gates 可引 bench_int8.full）
        row = pick_gen(b, f"{args.run_name}/gridhead_int8")
        if row:
            summary["bench_int8"] = {"full": row["full"], "total_err": row["total_err"]}
        # v45：human_best 基座 int8（部署口径统一到这里——双口径分叉的部署侧答案）
        row = pick_gen(b, f"{args.run_name}/gridhead_int8_hb")
        if row:
            summary["bench_int8_hb"] = {"full": row["full"], "total_err": row["total_err"]}

    # int8/体积/延迟（int8 那次 export_report.json 未改名，字段落 auto_summary）
    rep = run_dir / "export_report.json"
    if rep.exists():
        e = json.loads(rep.read_text(encoding="utf-8"))
        exp = {"fp32_ms": e.get("fp32_ms_desktop"), "int8_ms": e.get("int8_ms_desktop"),
               "nano_ms": e.get("nano_ms_desktop"), "int8_mb": e.get("int8_mb"),
               "fp32_mb": e.get("fp32_mb"),
               "int8_agree_render": e.get("int8_vs_fp32_agree"),
               "int8_agree_real": e.get("int8_vs_fp32_agree_real")}
        summary["export"] = {k: v for k, v in exp.items() if v is not None}

    # 判据评估（preset.gates；metric 取不到 → 缺数据）
    gates = []
    for g in preset.get("gates", []):
        val = deep_get(summary, g["metric"])
        base = deep_get(baseline_summary, g["metric"]) if (baseline_summary := _load_baseline(args)) else None
        if val is None:
            gates.append({**g, "actual": None, "verdict": "缺数据"})
            continue
        ok = (val >= g["value"]) if g["op"] == ">=" else (val <= g["value"])
        gates.append({**g, "actual": val,
                      "verdict": "PASS" if ok else "FAIL",
                      "baseline": base,
                      "delta": (round(val - base, 4) if isinstance(base, (int, float)) else None)})
    summary["gates"] = gates
    all_pass = bool(gates) and all(g["verdict"] == "PASS" for g in gates)

    (run_dir / "auto_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report_md(summary, preset, args, log)
    # v49 软门修正：DONE 标记反映 gates（此前只查 rc——v47 两门 FAIL 仍写 DONE，
    # "训练成功"≠"达标"，不退步承诺落到标记语义上；标记内容自带 verdict 说明）。
    # parse_train_log 从不返回 "failed" 键，旧判据里的 tl.get("failed") 是死代码（红队指出）
    ok = train_rc == 0
    reached = ok and all_pass
    (run_dir / ("DONE" if reached else "FAILED")).write_text(
        f"rc={train_rc} gates_all_pass={all_pass} "
        f"verdict={'达标' if reached else ('gates未全过' if ok else '训练失败')} "
        f"time={summary['time']}\n", encoding="utf-8")
    log(f"汇总完成: auto_summary.json + 自动报告.md（判据 {'全过' if all_pass else '未全过/缺数据'}）")
    return reached


def _load_baseline(args) -> dict | None:
    p = args.baseline or (args.preset_data.get("baseline", {}) or {}).get("path", "")
    if p and Path(p).exists():
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def write_report_md(summary: dict, preset: dict, args, log: RunLog):
    run_dir = HERE / "runs" / args.run_name
    L: list[str] = []
    L.append(f"# {args.run_name} 训练自动报告\n")
    L.append(f"- 时间：{summary['time']}　精度：{summary['precision']}　"
             f"训练 rc：{summary['train_rc']}")
    tl = summary.get("train", {})
    if tl.get("final_step"):
        L.append(f"- 训练：停止于 step {tl['final_step']}，best full_err0={tl.get('best_err0')}"
                 + ("（bench 10/10 提前达成）" if tl.get("early_stop_bench_full") else ""))
    L.append(f"- 选点 step：gen {summary['select'].get('gen')} / "
             f"human {summary['select'].get('human')} / bench {summary['select'].get('bench')}"
             f" / deploy {summary['select'].get('deploy')}\n")
    if summary.get("health"):
        L.append(f"- ⚠ 健康红旗：{summary['health']['redflag']}")
    # v49 报告诚实化（红队 #4/#1）：选点同点时多口径快照是同一权重，md5 注记
    # 防"双口径"重复计数（v48 实证：三口径 fp32 md5 相同，6 行数据实为 2 份）
    _md5s = {}
    for _n in ("gen", "human", "bench", "deploy"):
        _p = run_dir / f"{_n}_fp32.onnx"
        if _p.exists():
            _md5s[_n] = hashlib.md5(_p.read_bytes()).hexdigest()[:8]
    if len(_md5s) > 1 and len(set(_md5s.values())) == 1:
        L.append(f"- ⚠ 同权重注记：{'/'.join(_md5s)} 快照 fp32 md5 相同（{_md5s['gen']}）"
                 "——报告各口径行实为同一权重，勿作独立证据")

    # 四轴泛化表
    axes = [a for a in ("style", "piece", "angle", "roboflow")
            if any(a in summary.get(k, {}) for k in ("gen", "gen_int8", "gen_nano"))]
    if axes:
        L.append("## 四轴泛化（occ / cell / err）\n")
        L.append("| 轴 | 本轮 gen_best fp32 | 本轮 int8 | nano |")
        L.append("|---|---|---|---|")
        for a in axes:
            def cell(src):
                v = summary.get(src, {}).get(a)
                return (f"{v['occ']:.4f} / {v['cell']:.4f} / {v['err']}" if v else "—")
            L.append(f"| {a} | {cell('gen')} | {cell('gen_int8')} | {cell('gen_nano')} |")
        L.append("")
        for alt, label in (("gen_human_fp32", "human_best fp32"), ("gen_bench_fp32", "bench_best fp32"),
                           ("gen_deploy", "deploy_best fp32")):
            if alt in summary:
                L.append(f"- {label} 四轴: " + "，".join(
                    f"{a} occ={summary[alt][a]['occ']}" for a in axes if a in summary[alt]))
        L.append("")

    # 26 帧 / bench（gen_best 主口径 + human_best/bench_best 选点口径）
    if "human26" in summary:
        h = summary["human26"]
        L.append("## 26 帧人工真值（gen_best fp32）\n")
        for dom, label in (("real17", "实物17"), ("app9", "App域9"), ("total", "合计26")):
            v = h.get(dom) or {}
            L.append(f"- {label}: cell={v.get('cell')} occ={v.get('occ')} 错格={v.get('err')}")
        L.append("")
    if "human26_human_best" in summary:
        v = summary["human26_human_best"].get("total") or {}
        L.append(f"- human_best 口径合计26: cell={v.get('cell')} occ={v.get('occ')} "
                 f"错格={v.get('err')}")
    if "human26_deploy" in summary:
        v = summary["human26_deploy"].get("total") or {}
        hd = summary["human26_deploy"].get("real17") or {}
        L.append(f"- **deploy_best 口径合计26: cell={v.get('cell')} occ={v.get('occ')} "
                 f"错格={v.get('err')}**（v48 终局复选=部署基座）")
        L.append(f"- deploy_best 口径实物17: cell={hd.get('cell')} occ={hd.get('occ')} "
                 f"错格={hd.get('err')}")
    # v50 评测双轨：v2 干净版（19 格标注修正）呈现
    if "human26v2_human_best" in summary:
        v = summary["human26v2_human_best"]
        vt = v.get("total") or {}
        hr = v.get("real17") or {}
        ha = v.get("app9") or {}
        L.append(f"\n## 26 帧 v2 干净版（19 格标注修正，主判据口径）\n")
        L.append(f"- **v2 实物17: cell={hr.get('cell')} occ={hr.get('occ')} 错格={hr.get('err')}**")
        L.append(f"- **v2 App域9: cell={ha.get('cell')} occ={ha.get('occ')} 错格={ha.get('err')}**")
        L.append(f"- **v2 合计26: cell={vt.get('cell')} occ={vt.get('occ')} 错格={vt.get('err')}**"
                 f"（对照 nano v2=2）")
    if "human26v2_deploy" in summary:
        v = summary["human26v2_deploy"].get("total") or {}
        L.append(f"- v2 deploy_best 口径合计26: 错格={v.get('err')}")
    if "bench" in summary:
        L.append(f"## bench 10 帧（gen_best fp32）\n- 全对 {summary['bench']['full']}/10，"
                 f"总错格 {summary['bench']['total_err']}")
    if "bench_bench_best" in summary:
        v = summary["bench_bench_best"]
        L.append(f"- bench_best 口径: 全对 {v['full']}/10，总错格 {v['total_err']}")
    if "bench_deploy" in summary:
        v = summary["bench_deploy"]
        L.append(f"- deploy_best 口径: 全对 {v['full']}/10，总错格 {v['total_err']}")
    if "bench_int8" in summary:
        v = summary["bench_int8"]
        L.append(f"- **int8 部署口径: 全对 {v['full']}/10，总错格 {v['total_err']}**"
                 f"（真机跑的就是它——v44 起上报告+设门，掩盖盲区已拆）")
    if "bench_int8_hb" in summary:
        v = summary["bench_int8_hb"]
        L.append(f"- **int8 部署口径（human_best 基座）: 全对 {v['full']}/10，总错格 {v['total_err']}**"
                 f"（v45 起部署基座统一 human_best——双口径分叉的部署侧答案）")
    if "human26_human_best" in summary or "bench_bench_best" in summary:
        L.append("")

    # 部署侧
    if "export" in summary:
        e = summary["export"]
        L.append("## 部署侧（int8，gen_best 基座）\n")
        L.append(f"- 体积 fp32 {e.get('fp32_mb','—')}MB → int8 {e.get('int8_mb','—')}MB；"
                 f"延迟 fp32 {e.get('fp32_ms','—')}ms / int8 {e.get('int8_ms','—')}ms / "
                 f"nano {e.get('nano_ms','—')}ms（桌面 CPU 口径）")
        L.append(f"- int8 一致率：渲染 {e.get('int8_agree_render','—')} / 真实 {e.get('int8_agree_real','—')}\n")

    # 判据 + baseline delta
    if summary.get("gates"):
        L.append("## 预注册判据\n")
        L.append("| 判据 | 阈值 | 实际 | 判定 | 与上轮 delta |")
        L.append("|---|---|---|---|---|")
        for g in summary["gates"]:
            act = g["actual"] if g["actual"] is not None else "—"
            delta = g.get("delta")
            dstr = f"{delta:+.4f}" if isinstance(delta, float) else "—"
            L.append(f"| {g['metric']}（{g.get('note','')}） | {g['op']}{g['value']} "
                     f"| {act} | {g['verdict']} | {dstr} |")
        L.append("")

    L.append("## 产物\n")
    L.append(f"- auto_summary.json（下轮 --baseline 对拍用）/ 自动报告.md / pipeline.log / watchdog.log")
    L.append(f"- 原始评测 json：eval_gen.json / eval_human26.json / eval_bench.json；"
             f"导出日志 post_*.log\n")
    (run_dir / "自动报告.md").write_text("\n".join(L), encoding="utf-8")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True, help="presets/<name>.json 配方名")
    ap.add_argument("--run-name", default="", help="runs/<name>；默认=preset 名")
    ap.add_argument("--baseline", default="", help="上轮 auto_summary.json，报告出 delta 列")
    ap.add_argument("--override", default="", help="覆盖 train 参数，如 'lr=1e-4,total-samples=4800'")
    ap.add_argument("--precision", default="", choices=["", "bf16", "fp32"],
                    help="覆盖训练精度（默认走 preset/train.py 的 bf16）")
    ap.add_argument("--no-guard", action="store_true", help="关闭 watchdog 自动终止（仅记录）")
    ap.add_argument("--no-n2n", action="store_true", help="跳过 net2net 自动初始化（如从 ckpt 续训）")
    ap.add_argument("--commit-warn", type=float, default=55.0)
    ap.add_argument("--commit-kill", type=float, default=58.0)
    ap.add_argument("--extra", default="", help="透传 train.py 的额外参数字符串")
    args = ap.parse_args()
    args.run_name = args.run_name or args.preset

    ppath = HERE / "presets" / f"{args.preset}.json"
    if not ppath.exists():
        raise SystemExit(f"preset 不存在: {ppath}")
    args.preset_data = json.loads(ppath.read_text(encoding="utf-8"))

    run_dir = HERE / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir)
    log(f"=== pipeline 启动 preset={args.preset} run={args.run_name} ===")
    ok = False
    try:
        cmd = precheck(args.preset_data, args, log)
        rc = run_train(cmd, run_dir, args, log)
        post_chain(args.preset_data, args, log)
        ok = summarize(args.preset_data, args, rc, log)
    except SystemExit as e:
        log(f"中止: {e}")
        (run_dir / "FAILED").write_text(f"precheck/abort: {e}\n", encoding="utf-8")
        raise
    finally:
        log(f"=== pipeline 结束（{'DONE' if ok else 'FAILED/中止'}）===")
        log.close()


if __name__ == "__main__":
    main()
