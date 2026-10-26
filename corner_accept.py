# -*- coding: utf-8 -*-
"""corner 系验收自动化（判据注册表驱动，v67 起）。

一条命令跑完评测并自动对照判据，输出一行 VERDICT——评测判读从
"人工读 ~150 行输出对表"压缩为"读 1 行"。判据唯一可信源=judges/*.json
（预注册，禁止擅改；新版本立项复制改名追加，--judges 指定）。

用法（cwd 任意，脚本内部固定 recog 为工作目录）：
  python corner_accept.py --pt runs/corner_v67/best.pt
  python corner_accept.py --pt ... --only qq,screen30      # 只跑指定评测
  python corner_accept.py --pt ... --judges judges/corner_v68.json

评测项与耗时：qq ~10s / screen30 ~2min / nano17 ~3min / photo ~2min / vid ~10s。
轻进程，可与训练并行（内存增量 <1GB）。
输出：stdout 逐项 [PASS/FAIL/SKIP] 一行 + 末行 VERDICT；
     详细结果 JSON 落 ../../.cache/reports/<run>_accept.json。
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # recog
CACHE = HERE.parent.parent / ".cache"           # phone/.cache（不入库）

# 各评测命令模板（{pt} 为权重路径占位）；全部轻进程
EVAL_CMDS = {
    "qq":       [sys.executable, "qq_basis_check.py", "{pt}"],
    "screen30": [sys.executable, "eval_screen30.py", "--corner-pt", "{pt}"],
    "nano17":   [sys.executable, "eval_nano17.py", "--corner-pt", "{pt}"],
    "photo":    [sys.executable, "corner_eval_pseudo.py", "--corner-pt", "{pt}"],
    "vid":      [sys.executable, "corner_eval_pseudo.py", "--corner-pt", "{pt}",
                 "--pseudo", "../../.cache/corner_vid_pool.npz"],
}
TIMEOUT_S = {"qq": 120, "screen30": 600, "nano17": 900, "photo": 600, "vid": 180}


def sh(cmd_key, pt):
    """跑一条评测，返回 (stdout 拼接, 错误或 None)。"""
    cmd = [c.replace("{pt}", str(pt)) if "{pt}" in c else c for c in EVAL_CMDS[cmd_key]]
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=TIMEOUT_S[cmd_key])
        if r.returncode != 0:
            return "", f"exit={r.returncode}: {r.stderr[-300:]}"
        return r.stdout, None
    except subprocess.TimeoutExpired:
        return "", f"超时 {TIMEOUT_S[cmd_key]}s"


def get(d, dotted, default=None):
    """按 'a.b' 点路径取嵌套 dict 值。"""
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


_T0 = 0.0  # 本次验收起跑时刻（collect_* 读落盘 JSON 前核对 mtime 防 stale）


def _fresh_json(path):
    """读 JSON 且 mtime 必须晚于本次验收起跑（-5s 容差）。

    v67 J4 翻案教训：子进程评测偶发秒退时落盘 JSON 是上一轮的旧数据，
    不校验 mtime 会把旧值当真值误判（旧 locate_fail=0 → 误 PASS）。
    stale → 返回 None，调用方走 stdout 兜底或 ERROR，绝不静默采信。
    """
    p = Path(path)
    if not p.exists() or p.stat().st_mtime < _T0 - 5:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------- 各评测的结果抽取（优先读落盘 JSON，stdout 兜底） ----------------

def collect_qq(pt):
    out, err = sh("qq", pt)
    m = re.search(r"平均 ([\d.]+)", out)
    r = {"qq_mean_px": float(m.group(1)) if m else 9999.0}
    return r, err


def collect_screen30(pt):
    out, err = sh("screen30", pt)
    d = _fresh_json(CACHE / "screen30_report.json") or {}
    if "env1" not in d:  # 落盘缺失/stale → stdout 兜底
        m1 = re.search(r"成功率 (\d+)/(\d+)", out)
        m3 = re.search(r"FAIL-wrong (\d+)", out)
        d = {"env1": {"det_ok": int(m1.group(1)) if m1 else 0, "n": int(m1.group(2)) if m1 else 30},
             "env3": {"fail_wrong": int(m3.group(1)) if m3 else 9999}}
    return d, err


def collect_nano17(pt):
    out, err = sh("nano17", pt)
    d = _fresh_json(CACHE / "nano17_eval.json") or {}
    if "summary" not in d:  # 落盘缺失/stale → stdout 兜底
        m = re.search(r"定位失败 (\d+)", out)
        d = {"summary": {"locate_fail": [None] * int(m.group(1)) if m else [None] * 99}}
    return {"locate_fail": len(d["summary"]["locate_fail"])}, err


def collect_pseudo(pt, key):
    out, err = sh(key, pt)
    dom = "vid" if key == "vid" else "photo"
    m = re.search(rf"\[{dom}\] 一致率 (\d+)/(\d+) = ([\d.]+)", out)
    if not m:
        return {"ok": 0, "n": 0, "rate": 0.0}, (err or f"[{dom}] 输出行未匹配")
    return {"ok": int(m.group(1)), "n": int(m.group(2)), "rate": float(m.group(3))}, err


def check_op(value, op, target):
    return {"<=": value <= target, ">=": value >= target,
            "==": value == target, "<": value < target, ">": value > target}[op]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="待验收权重（best.pt / .onnx）")
    ap.add_argument("--judges", default=str(HERE / "judges/corner_v67.json"))
    ap.add_argument("--only", default=None, help="逗号分隔只跑指定评测（qq,screen30,nano17,photo,vid）")
    args = ap.parse_args()

    J = json.loads(Path(args.judges).read_text(encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    run_name = Path(args.pt).parent.name
    global _T0
    _T0 = time.time()

    results, details, fails, skipped = {}, {}, [], []
    t0 = time.time()
    print(f"[corner_accept] 权重 {args.pt} 判据 {Path(args.judges).name}", flush=True)
    for rule in J["rules"]:
        ev = rule["eval"]
        if ev == "manual" or (only and ev not in only):
            skipped.append(rule["id"])
            print(f"[{rule['id']} {rule['name']}] SKIP（{'真机观测项' if ev == 'manual' else '未选'}）",
                  flush=True)
            continue
        if ev == "qq":
            r, err = collect_qq(args.pt)
        elif ev == "screen30":
            r, err = collect_screen30(args.pt)
        elif ev == "nano17":
            r, err = collect_nano17(args.pt)
        else:
            r, err = collect_pseudo(args.pt, ev)
        if err and not r:
            fails.append(rule["id"])
            details[rule["id"]] = {"error": err}
            print(f"[{rule['id']} {rule['name']}] ERROR {err}", flush=True)
            continue
        results[rule["id"]] = r
        val = get(r, rule["metric"])
        ok = val is not None and check_op(val, rule["op"], rule["target"])
        if not ok:
            fails.append(rule["id"])
        mark = "PASS" if ok else "FAIL"
        print(f"[{rule['id']} {rule['name']}] {mark}: {rule['metric']}={val} "
              f"{rule['op']} {rule['target']}（基线 {rule['baseline']}）"
              + (f" | warn: {err}" if err else ""), flush=True)
        # screen30 双规则共用一次运行，结果留档去重
        details[rule["id"]] = {rule["metric"]: val, "verdict": mark}

    req_fail = [i for i in fails
                if next(x for x in J["rules"] if x["id"] == i)["required"]]
    verdict = "PASS" if not req_fail else "FAIL"
    summary = {
        "pt": str(args.pt), "judges": Path(args.judges).name,
        "verdict": verdict, "required_fails": req_fail,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "results": details, "skipped": skipped,
    }
    print(f"\nVERDICT: {verdict}"
          + (f" 必判未达={req_fail}" if req_fail else f"（全部必判项通过，耗时 {summary['elapsed_min']}min）"),
          flush=True)

    out_dir = CACHE / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{run_name}_accept.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"详细结果已存 {out}", flush=True)


if __name__ == "__main__":
    main()
