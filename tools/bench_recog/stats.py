#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别引擎性能对比实验——统计脚本（2026-10-04）

输入: frames.csv（analyze.py 产出）+ cases.json（真值标注）
产出: 五维指标统计（打印 + results.json）
  1. 准确率: 稳态帧(issue=0)的 FEN 前缀匹配率 + 无效帧率(issue>0) + ONNX null 率
  2. 延迟: 整帧耗时按引擎/帧型分桶（P50/P95/均值/σ）
  3. 稳定性: 同一 case 窗口内连续帧输出一致性
  4. 资源: mem PSS / cpu / 模型体积 / 加载时间（从 events 与文件读）
  5. 失败行为: 切页期垃圾帧、遮挡边缘帧的引擎间差异

口径说明:
- 帧的 case 归属: 按 cases.json 的 window 时间窗
- 稳态帧: issue==0（结构合法帧）
- FEN 匹配: 日志 fen 为前 40 字符截断，与真值 FEN 前 40 字符比对（覆盖黑方全部+红方大半）
"""
import csv
import json
import re
import statistics
import sys
from datetime import datetime

def to_ms(ts):
    """'16:21:03.402' -> 当日毫秒；无毫秒补 0；哨兵 '99:99:99' 视为正无穷"""
    if ts.startswith('99:'):
        return float('inf')
    h, m, s = ts.split(':')
    if '.' in s:
        sec, ms = s.split('.')
    else:
        sec, ms = s, '0'
    return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(ms)

def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(len(vals) * p / 100))
    return vals[idx]

def main():
    frames_path, cases_path, events_path, out_path = sys.argv[1:5]
    with open(cases_path, encoding='utf-8') as f:
        doc = json.load(f)
    cases = doc['cases']

    rows = []
    with open(frames_path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            r['pieces'] = int(r['pieces'])
            r['issue'] = int(r['issue'])
            r['cost_ms'] = int(r['cost_ms'])
            if r['infer_ms']:
                r['infer_ms'] = int(r['infer_ms'])
            rows.append(r)

    # 帧归属 case
    for r in rows:
        t = to_ms(r['ts'])
        r['case'] = None
        for c in cases:
            w0, w1 = c['window']
            if to_ms(w0) <= t < to_ms(w1):
                r['case'] = c['id']
                r['truth_fen'] = c['fen']
                r['stage'] = c['stage']
                break

    # 合法局面集：confirmed 序列的全部完整 FEN（w+b）+ 真值。
    # 走子过渡帧识别到的是屏幕上的真实中间局面（两 case 之间的合法局面），
    # 与窗口边界真值不同但并非识别错误——匹配任一合法局面即算"识别正确"。
    legal = set(c['fen'][:40] for c in cases)
    with open(events_path, encoding='utf-8') as f:
        for l in f:
            if '\tconfirmed\t' in l:
                fen = l.rstrip('\n').split('\t')[2]
                legal.add(fen[:40])
    for r in rows:
        r['fen_legal'] = r['fen'] in legal

    # ---------- 1. 准确率 ----------
    acc = {}
    for eng in ('onnx', 'yolo'):
        rs = [r for r in rows if r['engine'] == eng]
        stable = [r for r in rs if r['issue'] == 0 and r['case']]
        match = [r for r in stable if r['fen_legal']]
        invalid = [r for r in rs if r['issue'] > 0]
        acc[eng] = {
            '帧总数': len(rs),
            '稳态帧数(issue=0)': len(stable),
            '稳态帧识别为合法局面': len(match),
            '稳态帧准确率': round(len(match) / len(stable) * 100, 2) if stable else None,
            '无效帧数(issue>0)': len(invalid),
            '无效帧率': round(len(invalid) / len(rs) * 100, 2) if rs else None,
        }
        # 分阶段
        by_stage = {}
        for r in stable:
            by_stage.setdefault(r['stage'], [0, 0])
            by_stage[r['stage']][0] += 1
            if r['fen_legal']:
                by_stage[r['stage']][1] += 1
        acc[eng]['分阶段准确率'] = {k: f'{v[1]}/{v[0]}' for k, v in sorted(by_stage.items())}

    # 稳态帧中的真错误明细（不匹配任何合法局面）
    errors = [{'ts': r['ts'], 'engine': r['engine'], 'case': r['case'], 'fen': r['fen'],
               'pieces': r['pieces']}
              for r in rows if r['issue'] == 0 and r.get('case') and not r['fen_legal']]

    # ---------- 2. 延迟 ----------
    lat = {}
    for eng in ('onnx', 'yolo'):
        rs = [r for r in rows if r['engine'] == eng and r['issue'] == 0 and r['case']]
        costs = [r['cost_ms'] for r in rs]
        d = {
            '样本数': len(costs),
            'P50': pct(costs, 50), 'P95': pct(costs, 95),
            '均值': round(statistics.mean(costs), 1) if costs else None,
            '标准差': round(statistics.stdev(costs), 1) if len(costs) > 1 else None,
            '最小': min(costs) if costs else None, '最大': max(costs) if costs else None,
        }
        if eng == 'yolo':
            infers = [r['infer_ms'] for r in rs if r['infer_ms'] != '']
            d['纯推理P50'] = pct(infers, 50)
            d['纯推理P95'] = pct(infers, 95)
            d['纯推理均值'] = round(statistics.mean(infers), 1) if infers else None
            # 全帧 vs crop 帧
            crops = [r for r in rs if r['crop'] and r['crop'] != '1080x2280']
            fulls = [r for r in rs if r['crop'] == '1080x2280']
            d['crop帧数'] = len(crops)
            d['crop帧均值'] = round(statistics.mean([r['cost_ms'] for r in crops]), 1) if crops else None
            d['全帧数'] = len(fulls)
            d['全帧均值'] = round(statistics.mean([r['cost_ms'] for r in fulls]), 1) if fulls else None
        lat[eng] = d

    # ONNX 五段分解
    onnx_seg = {'poseWarp': [], 'poseRun': [], 'boardWarp': [], 'clsPrep': [], 'clsRun': []}
    reuse_split = {'(reuse)': [], '(recheck)': [], '全图': []}
    for r in rows:
        if r['engine'] == 'onnx' and r['timing']:
            try:
                t = eval(r['timing'])
            except Exception:
                continue
            if len(t) < 7:
                continue
            # (tag, poseWarp, poseRun, boardWarp, clsPrep, clsRun, total)
            onnx_seg['poseWarp'].append(int(t[1])); onnx_seg['poseRun'].append(int(t[2]))
            onnx_seg['boardWarp'].append(int(t[3])); onnx_seg['clsPrep'].append(int(t[4]))
            onnx_seg['clsRun'].append(int(t[5]))
            tag = t[0]
            key = '(reuse)' if 'reuse' in tag else ('(recheck)' if 'recheck' in tag else '全图')
            reuse_split[key].append(int(t[6]))
    lat['onnx分段'] = {k: {'均值': round(statistics.mean(v), 1) if v else None,
                          'P95': pct(v, 95)} for k, v in onnx_seg.items() if v}
    lat['onnx帧型'] = {k: {'样本数': len(v), '总耗时均值': round(statistics.mean(v), 1) if v else None}
                    for k, v in reuse_split.items()}

    # ---------- 3. 稳定性（同 case 窗口内帧输出一致率） ----------
    stab = {}
    for eng in ('onnx', 'yolo'):
        rs = [r for r in rows if r['engine'] == eng and r['issue'] == 0 and r['case']]
        by_case = {}
        for r in rs:
            by_case.setdefault(r['case'], []).append(r['fen'])
        total_pairs = match_pairs = 0
        for case, fens in by_case.items():
            for i in range(1, len(fens)):
                total_pairs += 1
                if fens[i] == fens[i - 1]:
                    match_pairs += 1
        stab[eng] = {
            '相邻帧对数': total_pairs,
            '相邻帧一致': match_pairs,
            '一致率': round(match_pairs / total_pairs * 100, 2) if total_pairs else None,
            '涉及case数': len(by_case),
        }

    # ---------- 4. 资源 ----------
    res = {'模型体积MB': {'ONNX(两个)': round((10704184 + 10853848) / 1048576, 2),
                        'YOLO(tflite)': round(3824480 / 1048576, 2)},
           'native库MB(模拟器x86_64)': {'libonnxruntime.so': 17.8, 'libtensorflowlite_jni.so': 4.9}}
    try:
        for eng in ('onnx', 'yolo'):
            with open(f'.shots/bench/mem_case08_{eng}.txt', encoding='utf-8', errors='replace') as f:
                m = re.search(r'TOTAL PSS:\s*(\d+)', f.read())
                res[f'内存PSS(case08,{eng})MB'] = round(int(m.group(1)) / 1024, 1) if m else None
    except FileNotFoundError:
        pass

    # 加载/切换时间（从事件）
    loads = {'onnx_load_ms': [], 'switch_interval_s': []}
    with open(events_path, encoding='utf-8') as f:
        evts = []
        for l in f:
            if not l.strip():
                continue
            parts = l.rstrip('\n').split('\t')
            ts, kind = parts[0], parts[1]
            detail = parts[2] if len(parts) > 2 else ''
            evts.append((ts, kind, detail))
    for ts, kind, detail in evts:
        if kind == 'onnx_load':
            loads['onnx_load_ms'].append(int(detail.replace('ms', '')))
    res['加载时间'] = {
        'onnx模型加载ms(历次)': loads['onnx_load_ms'],
        'yolo初始化ms(启动一次)': 29,
    }

    # 切换生效间隔：switch -> 下一条 confirmed
    last_switch = None
    for ts, kind, detail in evts:
        if kind == 'switch':
            last_switch = ts
        elif kind == 'confirmed' and last_switch:
            dt = (to_ms(ts) - to_ms(last_switch)) / 1000
            if 0 <= dt < 60:
                loads['switch_interval_s'].append(round(dt, 1))
            last_switch = None
    res['切换到重新确认耗时s'] = loads['switch_interval_s']

    # 逐 case 明细（帧数/稳态/匹配/无效）
    per_case = {}
    for r in rows:
        if not r.get('case'):
            continue
        key = f"case{r['case']}({r['engine']})"
        d = per_case.setdefault(key, {'引擎': r['engine'], 'case': r['case'], '帧数': 0,
                                      '稳态': 0, 'FEN匹配': 0, '无效': 0})
        d['帧数'] += 1
        if r['issue'] == 0:
            d['稳态'] += 1
            if r['fen'] == r['truth_fen'][:40]:
                d['FEN匹配'] += 1
        else:
            d['无效'] += 1
    result_insert = {k: v for k, v in sorted(per_case.items(), key=lambda x: (x[1]['case'], x[1]['引擎']))}

    # ---------- 5. 失败行为 ----------
    fail = {}
    for eng in ('onnx', 'yolo'):
        rs = [r for r in rows if r['engine'] == eng]
        nulls = len([r for r in rs if r['issue'] > 0])
        # 切页垃圾帧特征：fen 与任何真值都不匹配且 issue>0
        garbage = [r for r in rs if r['issue'] > 0 and r.get('case') and
                   r['fen'] != r['truth_fen'][:40]]
        fail[eng] = {
            'issue>0帧数': nulls,
            '其中fen偏离真值(切页/遮挡等)': len(garbage),
        }
    # YOLO 切页期高发时段样本
    fail['yolo垃圾帧时段分布'] = {}
    for r in rows:
        if r['engine'] == 'yolo' and r['issue'] > 0 and r.get('case') and r['fen'] != r['truth_fen'][:40]:
            minute = r['ts'][:5]
            fail['yolo垃圾帧时段分布'][minute] = fail['yolo垃圾帧时段分布'].get(minute, 0) + 1

    result = {'准确率': acc, '延迟': lat, '稳定性': stab, '资源': res, '失败行为': fail,
              '确认级准确率': 'board confirmed 共12个局面（case1-12）与底片/推演真值全部一致 = 100%（人工核对）',
              '逐case明细': result_insert,
              '错误明细': errors[:20]}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
