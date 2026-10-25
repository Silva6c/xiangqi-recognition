# -*- coding: utf-8 -*-
"""爬取真实中国象棋棋盘照片（数据引擎实拍域扩展，v16~v17 计划 A2）。

双引擎自动回退：
- 百度图片 acjson JSON 接口（国内直连，不需要代理）
- 必应图片 async HTML 接口（cn.bing.com 直连；失败时切 www.bing.com 走本机代理）

产出：data/photo_crawl/raw/<安全关键词>_<序号>.jpg + manifest.json（来源/关键词/尺寸）
去重：内容 MD5 + 8x8 aHash 汉氏距离（自实现，不引第三方依赖）
过滤：PIL 可解码、最小边 ≥ --min-size、宽高比 0.5~2.0
限速：请求间 0.6~1.2s 随机退避，避免触发反爬
合规：仅用于本仓库模型内部训练，不进 git、不对外分发原始图片

用法：
  python crawl_photos.py --smoke                 # 冒烟：2 个关键词各 8 张，验证接口
  python crawl_photos.py --max-per-kw 70        # 全量爬取
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

HERE = Path(__file__).parent
PROXY = "http://127.0.0.1:7897"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# 关键词矩阵：覆盖棋摊/对弈/俯拍/特写多视角
KEYWORDS = [
    "象棋棋摊", "街头象棋", "路边象棋", "象棋实拍", "象棋对弈照片",
    "象棋棋盘俯拍", "公园象棋", "大爷下象棋", "象棋比赛现场", "中国象棋棋盘",
    "象棋摊照片", "下棋老人照片", "街头棋局", "象棋棋子特写", "象棋残局摆棋",
    "象棋比赛俯拍",
]


def ahash(img: Image.Image, size: int = 8) -> tuple:
    """8x8 平均哈希：用于近重复图剔除。"""
    g = img.convert("L").resize((size, size), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    return tuple(p > avg for p in px)


def hamming(a: tuple, b: tuple) -> int:
    return sum(x != y for x, y in zip(a, b))


def baidu_urls(kw: str, pn: int, sess: requests.Session) -> list[str]:
    """百度图片 acjson 接口：返回一页缩略/中图 URL。"""
    params = dict(tn="resultjson_com", ipn="rj", ct=201326592, fp="result",
                  word=kw, queryWord=kw, cl=2, lm=-1, ie="utf-8", oe="utf-8",
                  pn=pn, rn=30, gsm=f"{pn:x}", face=0)
    r = sess.get("https://image.baidu.com/search/acjson", params=params,
                 timeout=15)
    r.raise_for_status()
    try:
        items = (json.loads(r.text).get("data") or [])
    except json.JSONDecodeError:
        # 百度偶发返回含非法 \x 转义的响应体，降级正则提取
        return re.findall(r'"(?:middleURL|thumbURL|hoverURL)":"(https?://[^"]+)"',
                          r.text)
    out = []
    for it in items:
        if not it:
            continue
        u = it.get("middleURL") or it.get("thumbURL") or it.get("hoverURL")
        if u:
            out.append(u)
    return out


def bing_urls(kw: str, first: int, sess: requests.Session) -> list[str]:
    """必应图片 async 接口：从 HTML 的 m="..." 属性里解析原图 murl。"""
    for base, use_proxy in (("https://cn.bing.com", False),
                            ("https://www.bing.com", True)):
        try:
            proxies = {"https": PROXY, "http": PROXY} if use_proxy else None
            r = sess.get(f"{base}/images/async",
                         params=dict(q=kw, first=first, count=35, mmasync=1),
                         timeout=15, proxies=proxies)
            r.raise_for_status()
            out = []
            for m in re.findall(r'm="([^"]+)"', r.text):
                try:
                    meta = json.loads(unicodedata.normalize("NFKC",
                                    m.encode().decode("unicode_escape")))
                except Exception:
                    continue
                u = meta.get("murl")
                if u and u.startswith("http"):
                    out.append(u)
            if out:
                return out
        except requests.RequestException:
            continue
    return []


def download(url: str, sess: requests.Session, use_proxy: bool) -> Image.Image | None:
    """下载并解码单图；失败返回 None。"""
    try:
        proxies = {"https": PROXY, "http": PROXY} if use_proxy else None
        r = sess.get(url, timeout=15, stream=True, proxies=proxies,
                     headers={"User-Agent": UA, "Referer": url[:60]})
        r.raise_for_status()
        data = r.raw.read(20 * 1024 * 1024, decode_content=True)
        return Image.open(__import__("io").BytesIO(data)).convert("RGB")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data/photo_crawl"))
    ap.add_argument("--keywords", default=None,
                    help="逗号分隔关键词覆盖(默认用内置矩阵)")
    ap.add_argument("--max-per-kw", type=int, default=70,
                    help="每个关键词入库上限（下载成功口径）")
    ap.add_argument("--min-size", type=int, default=400, help="最小边长过滤")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：2 词 x 8 张")
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # 历史版本曾以系统默认编码读取导致中途崩溃，尽量恢复可解析前缀
            txt = manifest_path.read_text(encoding="utf-8", errors="ignore")
            for cut in range(len(txt), 0, -1):
                try:
                    manifest = json.loads(txt[:cut] + "]")
                    break
                except json.JSONDecodeError:
                    continue
            print(f"manifest 已损坏，恢复出 {len(manifest)} 条")
    seen_md5 = {m["md5"] for m in manifest}
    seen_hash = [tuple(m["ahash"]) for m in manifest]

    kws = (args.keywords.split(",") if args.keywords else KEYWORDS)
    kws = kws[:2] if args.smoke else kws
    per_kw = 8 if args.smoke else args.max_per_kw

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    stats = {"ok": 0, "dup": 0, "bad": 0, "fail": 0}

    for kw in kws:
        safe = re.sub(r"\s+", "", kw)
        got = sum(1 for m in manifest if m["kw"] == kw)
        page = 0
        while got < per_kw and page < per_kw * 2:
            urls = baidu_urls(kw, page * 30, sess)
            if len(urls) < 5:
                urls = urls or bing_urls(kw, page * 35, sess)
            if not urls:
                urls = bing_urls(kw, page * 35, sess)
            page += 1
            if not urls:
                time.sleep(2)
                if page > 3:
                    break
                continue
            for u in urls:
                if got >= per_kw:
                    break
                is_baidu = "baidu.com" in u or "bdimg" in u or "gtimg" in u
                img = download(u, sess, use_proxy=not is_baidu)
                time.sleep(random.uniform(0.4, 1.0))
                if img is None:
                    stats["fail"] += 1
                    continue
                w, h = img.size
                if min(w, h) < args.min_size or not 0.5 <= w / h <= 2.0:
                    stats["bad"] += 1
                    continue
                raw = __import__("io").BytesIO()
                img.save(raw, "JPEG", quality=92)
                data = raw.getvalue()
                md5 = hashlib.md5(data).hexdigest()
                hs = ahash(img)
                if md5 in seen_md5 or any(hamming(hs, s) < 6 for s in seen_hash):
                    stats["dup"] += 1
                    continue
                name = f"{safe}_{len(manifest):04d}.jpg"
                (raw_dir / name).write_bytes(data)
                manifest.append({"file": name, "kw": kw, "src": u,
                                 "w": w, "h": h, "md5": md5,
                                 "ahash": list(hs)})
                seen_md5.add(md5)
                seen_hash.append(hs)
                got += 1
                stats["ok"] += 1
        print(f"[{kw}] 入库 {got} 张（累计 {stats}）", flush=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False),
                                 encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"完成：{stats}，manifest {len(manifest)} 条 → {manifest_path}")


if __name__ == "__main__":
    main()
