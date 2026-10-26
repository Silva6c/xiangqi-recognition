#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别引擎性能对比实验——logcat 解析脚本（2026-10-04）

把 bench_logcat.txt 解析为逐帧记录 frames.csv：
- 按 "switch engine" 行切分段落，段落内引擎已知
- rec(onnx)/rec(yolo) 行 = 每次识别一帧（fen 为日志截断的前 40 字符）
- yolo: infer= 行与 rec(yolo) 同毫秒配对，补充推理细节
- timing 行与 rec(onnx) 同毫秒配对，补充 ONNX 五段分解
- board confirmed / 新局开始 / 引擎加载行作为事件列

用法: python analyze.py <logcat文件> <输出csv>
"""
import re
import csv
import sys

def parse_ts(ts):
    """'10-03 16:21:03.402' -> 归一化毫秒数（按当天内排序用）"""
    m = re.match(r'(\d+)-(\d+) (\d+):(\d+):(\d+)\.(\d+)', ts)
    if not m:
        return None
    mon, day, h, mi, s, ms = map(int, m.groups())
    return ((mon * 31 + day) * 24 * 3600 + h * 3600 + mi * 60 + s) * 1000 + ms

def main():
    log_path, out_path = sys.argv[1], sys.argv[2]
    rows = []          # 逐帧记录
    events = []        # 事件行（switch/confirmed/load）
    engine = 'onnx'    # 日志起点时的引擎（实验机默认 onnx）
    seg_id = 0         # 段号（每次 switch +1）
    yolo_detail = {}   # 毫秒 -> infer 细节
    onnx_timing = {}   # 毫秒 -> timing 细节

    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.match(r'([\d-]+ [\d:.]+) [DIWE]/([\w.]+)\(\s*\d+\):\s*(.*)', line)
            if not m:
                continue
            ts_raw, tag, msg = m.groups()
            t = parse_ts(ts_raw)
            if t is None:
                continue
            ts = ts_raw.split(' ')[1]

            if 'switch engine:' in msg:
                mm = re.search(r'switch engine: (\w+) -> (\w+)', msg)
                if mm:
                    seg_id += 1
                    events.append((ts, 'switch', f'{mm.group(1)}->{mm.group(2)}'))
                    engine = mm.group(2)
                continue
            if 'board confirmed:' in msg:
                fen = msg.split('board confirmed:')[1].strip()
                events.append((ts, 'confirmed', fen))
                continue
            if '新局开始' in msg:
                mm = re.search(r'新局开始 (fen=\S+)', msg)
                events.append((ts, 'new_game', mm.group(1) if mm else ''))
                continue
            if 'onnx models loaded in' in msg:
                mm = re.search(r'loaded in (\d+)ms', msg)
                events.append((ts, 'onnx_load', mm.group(1) + 'ms'))
                continue
            if 'yolo detector:' in msg and 'init=' in msg:
                mm = re.search(r'init=(\d+)ms', msg)
                events.append((ts, 'yolo_init', mm.group(1) + 'ms'))
                continue

            # YOLO 推理细节行（与 rec(yolo) 同毫秒）
            if 'yolo: infer=' in msg:
                mm = re.search(r'infer=(\d+)ms crop=(\d+x\d+) dets=(\d+) pieces=(\d+) avg=([\d.]+)', msg)
                if mm:
                    yolo_detail[t] = mm.groups()
                continue
            # ONNX 分段计时行（与 rec(onnx) 同毫秒）
            if 'timing' in msg and 'poseWarp=' in msg:
                mm = re.search(r'timing(\([^)]*\))?: poseWarp=(\d+)ms poseRun=(\d+)ms boardWarp=(\d+)ms clsPrep=(\d+)ms clsRun=(\d+)ms total=(\d+)ms', msg)
                if mm:
                    onnx_timing[t] = (mm.group(1) or '', *mm.groups()[1:])
                continue

            if 'onnx recognize -> null' in msg:
                mm = re.search(r'cost=(\d+)', msg)
                events.append((ts, 'onnx_null', mm.group(1) + 'ms' if mm else ''))
                continue
            if 'yolo detect failed' in msg:
                events.append((ts, 'yolo_fail', ''))
                continue
            if '遮挡期间识别结果无效' in msg:
                events.append((ts, 'fuse_drop', ''))
                continue

            # 识别帧行
            mo = re.search(r'rec\(onnx\): pieces=(\d+) orient=(\w+) issue=(\d+) fen=(\S+) (\d+)ms', msg)
            my = re.search(r'rec\(yolo\): pieces=(\d+) orient=(\w+) issue=(\d+) dropped=(\d+) fen=(\S+) (\d+)ms', msg)
            if mo:
                pieces, orient, issue, fen, cost = mo.groups()
                rows.append(dict(ts=ts, tms=t, seg=seg_id, engine='onnx', pieces=int(pieces),
                                 orient=orient, issue=int(issue), fen=fen, cost_ms=int(cost),
                                 dropped='', infer_ms='', crop='', avg='',
                                 timing=repr(onnx_timing.get(t, ''))))
            elif my:
                pieces, orient, issue, dropped, fen, cost = my.groups()
                d = yolo_detail.get(t, ('', '', '', '', ''))
                rows.append(dict(ts=ts, tms=t, seg=seg_id, engine='yolo', pieces=int(pieces),
                                 orient=orient, issue=int(issue), fen=fen, cost_ms=int(cost),
                                 dropped=dropped, infer_ms=d[0], crop=d[1], avg=d[3],
                                 timing=''))

    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['ts', 'tms', 'seg', 'engine', 'pieces', 'orient',
                                          'issue', 'fen', 'cost_ms', 'dropped', 'infer_ms',
                                          'crop', 'avg', 'timing'])
        w.writeheader()
        w.writerows(rows)

    with open(out_path.replace('.csv', '_events.txt'), 'w', encoding='utf-8') as f:
        for ts, kind, detail in events:
            f.write(f'{ts}\t{kind}\t{detail}\n')

    print(f'帧记录 {len(rows)} 条，事件 {len(events)} 条 -> {out_path}')
    # 分段概览
    segs = {}
    for r in rows:
        segs.setdefault(r['seg'], dict(engine=r['engine'], n=0))
        segs[r['seg']]['n'] += 1
    for s in sorted(segs):
        print(f'  段{s}: 引擎={segs[s]["engine"]} 帧数={segs[s]["n"]}')

if __name__ == '__main__':
    main()
