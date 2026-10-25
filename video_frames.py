# -*- coding: utf-8 -*-
"""B 站象棋实拍视频下载 + ffmpeg 抽帧（数据引擎实拍域扩展，v16~v17 计划 A3）。

流程：yt-dlp bilisearch 搜索并下载（≤720p、单视频 ≤20 分钟）→ ffmpeg 每 5s 抽帧
→ 抽帧后删除视频文件只留帧（控磁盘）。

产出：data/photo_crawl/video_frames/v<视频id>_<序号>.jpg + video_manifest.json
合规：仅用于本仓库模型内部训练，不进 git、不对外分发。

用法：python video_frames.py --per-kw 3 --max-frames 400
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent

# 关键词：偏实拍/街头/教学场景（俯拍与斜拍都有）
KW = ["街头象棋", "象棋教学 棋盘", "公园象棋对弈", "象棋实战讲解", "象棋残局"]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-800:])
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data/photo_crawl"))
    ap.add_argument("--per-kw", type=int, default=3, help="每个关键词下载视频数")
    ap.add_argument("--max-frames", type=int, default=500, help="总抽帧上限")
    args = ap.parse_args()

    out_dir = Path(args.out)
    vid_dir = out_dir / "videos"
    frm_dir = out_dir / "video_frames"
    vid_dir.mkdir(parents=True, exist_ok=True)
    frm_dir.mkdir(parents=True, exist_ok=True)
    mani_path = out_dir / "video_manifest.json"
    mani = json.loads(mani_path.read_text("utf-8")) if mani_path.exists() else {}

    total = sum(len(v) for v in mani.values())
    for kw in KW:
        if total >= args.max_frames:
            break
        try:
            # 搜索下载：≤720p、≤20 分钟，只下视频轨
            sh(["yt-dlp", "-f", "bv*[height<=720]", "--match-filter",
                "duration<1200", "--no-part", "--restrict-filenames",
                "-o", str(vid_dir / "%(id)s.%(ext)s"),
                f"bilisearch{args.per_kw}:{kw}"])
        except RuntimeError as e:
            print(f"[{kw}] 下载失败: {e}", flush=True)
        for vf in sorted(vid_dir.iterdir()):
            vid = vf.stem
            if vid in mani:
                continue
            try:
                out_pat = str(frm_dir / f"v{vid}_%04d.jpg")
                sh(["ffmpeg", "-y", "-i", str(vf), "-vf", "fps=1/5",
                    "-q:v", "3", "-frames:v",
                    str(args.max_frames - total), out_pat])
                frames = sorted(frm_dir.glob(f"v{vid}_*.jpg"))
                mani[vid] = {"kw": kw, "video": vf.name, "frames": len(frames)}
                total += len(frames)
                print(f"[{kw}] {vid}: 抽 {len(frames)} 帧（累计 {total}）",
                      flush=True)
            except RuntimeError as e:
                print(f"[{kw}] {vid} 抽帧失败: {e}", flush=True)
                mani[vid] = {"kw": kw, "video": vf.name, "frames": 0,
                             "error": True}
            finally:
                vf.unlink(missing_ok=True)  # 控磁盘：抽完即删视频
                mani_path.write_text(json.dumps(mani, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"完成：{total} 帧 → {frm_dir}")


if __name__ == "__main__":
    main()
