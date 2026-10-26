# -*- coding: utf-8 -*-
"""开源中文字体池抓取脚本（自研识别引擎渲染器的原料）。

用途：下载 OFL/免费商用授权的中文字体到 fonts/ 目录，供 renderer.py
随机选用字体渲染棋子文字。字体多样性的目标：覆盖象棋 App 常见的
楷/隶/宋/黑/圆/手写/美术风格（上游弱点情报：样式多样性不足是痛点）。

来源策略（docs/自研识别引擎方针.md §三 D 盘硬约束）：
- GitHub 源（google/fonts、adobe-fonts、lxgw）走本机代理（注册表实测 127.0.0.1:7897）
- 国内源（阿里/小米 OSS）直连不走代理
- 全部落盘到 phone/tools/recog/fonts/（已 gitignore，重跑本脚本可再生）

用法：python fetch_fonts.py [--fonts-dir PATH]
产出：fonts/*.ttf|*.otf + fonts/manifest.json（名称/来源/大小/校验）
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

# 本机代理（来自注册表 ProxyServer 实测，非猜测；仅用于 GitHub 等被墙源）
PROXY = "http://127.0.0.1:7897"
GITHUB_API = "https://api.github.com"
UA = {"User-Agent": "xiangqi-recognition-font-fetch/1.0"}

GOOGLE_FONTS = "https://github.com/google/fonts/raw/main/ofl"


def github_via_proxy(url: str, timeout: int = 120) -> bytes:
    """经代理抓取（GitHub 等被墙源）"""
    handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(url, headers=UA)
    with opener.open(req, timeout=timeout) as r:
        return r.read()


def direct(url: str, timeout: int = 120) -> bytes:
    """直连抓取（国内源，不走代理）"""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def lxgw_latest_regular() -> str:
    """解析霞鹜文楷最新 release 的 Regular 资产直链（tag 会变，动态解析）"""
    data = json.loads(github_via_proxy(f"{GITHUB_API}/repos/lxgw/LxgwWenKai/releases/latest"))
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if name in ("LXGWWenKai-Regular.ttf", "LXGWWenKai-Medium.ttf"):
            return asset["browser_download_url"]
    raise RuntimeError("lxgw release 中未找到 Regular/Medium 资产")


def lxgw_latest_medium() -> str:
    """解析霞鹜文楷最新 release 的 Medium 资产直链（字重多样性）"""
    data = json.loads(github_via_proxy(f"{GITHUB_API}/repos/lxgw/LxgwWenKai/releases/latest"))
    for asset in data.get("assets", []):
        if asset.get("name") == "LXGWWenKai-Medium.ttf":
            return asset["browser_download_url"]
    raise RuntimeError("lxgw release 中未找到 Medium 资产")


# (文件名, 抓取器, 说明) —— 抓取器为惰性函数，返回 (url, fetch_fn)
def build_targets() -> list:
    t = []

    def gh(name: str, path: str, desc: str):
        t.append((name, lambda: (f"{GOOGLE_FONTS}/{path}", github_via_proxy), desc))

    # 谷歌开源中文字体（楷/隶/手写/美术风格为主，正是棋子字形的重点分布）
    gh("MaShanZheng-Regular.ttf", "mashanzheng/MaShanZheng-Regular.ttf", "楷书毛笔")
    gh("ZCOOLKuaiLe-Regular.ttf", "zcoolkuaile/ZCOOLKuaiLe-Regular.ttf", "美术快乐体")
    gh("ZCOOLXiaoWei-Regular.ttf", "zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf", "小薇宋体风")
    gh("ZCOOLQingKeHuangYou-Regular.ttf", "zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf", "黑体美术")
    gh("LongCang-Regular.ttf", "longcang/LongCang-Regular.ttf", "手写楷")
    gh("LiuJianMaoCao-Regular.ttf", "liujianmaocao/LiuJianMaoCao-Regular.ttf", "草书手写")
    gh("ZhiMangXing-Regular.ttf", "zhimangxing/ZhiMangXing-Regular.ttf", "手写行书")
    gh("NotoSansSC-Regular.ttf", "notosanssc/NotoSansSC%5Bwght%5D.ttf", "思源黑体(可变字重默认Regular)")
    gh("NotoSerifSC-Regular.ttf", "notoserifsc/NotoSerifSC%5Bwght%5D.ttf", "思源宋体(可变字重默认Regular)")

    # Adobe 思源系列（SubsetOTF CN 子集，约 9MB/个）
    t.append(("SourceHanSansCN-Regular.otf",
              lambda: ("https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/CN/SourceHanSansCN-Regular.otf",
                       github_via_proxy), "思源黑体 CN"))
    t.append(("SourceHanSerifCN-Bold.otf",
              lambda: ("https://github.com/adobe-fonts/source-han-serif/raw/release/SubsetOTF/CN/SourceHanSerifCN-Bold.otf",
                       github_via_proxy), "思源宋体 CN Bold"))

    # 霞鹜文楷（楷体感强，很多象棋 App 用楷体系）
    t.append(("LXGWWenKai-Regular.ttf", lambda: (lxgw_latest_regular(), github_via_proxy), "霞鹜文楷"))
    t.append(("LXGWWenKai-Medium.ttf", lambda: (lxgw_latest_medium(), github_via_proxy), "霞鹜文楷 Medium"))

    # v18 扩池：楷/圆/黑多字重，拉大字形风格覆盖（holdout 三字体仍是禁区，不新增宋体系）
    t.append(("SourceHanSansCN-Bold.otf",
              lambda: ("https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/CN/SourceHanSansCN-Bold.otf",
                       github_via_proxy), "思源黑体 CN Bold"))
    t.append(("SourceHanSansCN-Medium.otf",
              lambda: ("https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/CN/SourceHanSansCN-Medium.otf",
                       github_via_proxy), "思源黑体 CN Medium"))
    t.append(("jf-openhuninn-2.1.ttf",
              lambda: ("https://github.com/justfont/open-huninn-font/releases/download/v2.1/jf-openhuninn-2.1.ttf",
                       github_via_proxy), "jf 粉圆 2.1（圆体，OFL）"))

    # 国内直连：阿里巴巴普惠体 3.0（Regular/Bold 双字重，OSS 官方位）
    for weight, path in (("Regular", "AlibabaPuHuiTi-3-55-Regular/AlibabaPuHuiTi-3-55-Regular.ttf"),
                         ("Bold", "AlibabaPuHuiTi-3-85-Bold/AlibabaPuHuiTi-3-85-Bold.ttf")):
        url = f"https://puhuiti.oss-cn-hangzhou.aliyuncs.com/AlibabaPuHuiTi-3/{path}"
        t.append((f"AlibabaPuHuiTi-3-{weight}.ttf", lambda u=url: (u, direct), "阿里普惠体3(国内直连)"))

    # v33 泛化轮扩池(目标 40+):更多字重/家族。禁区不变——GEN_HOLDOUT 三体与
    # 评测专属 8 体(系统字体 FZSTK/STCAIYUN/STHUPO/STLITI/STSONG/STXINGKA/STXINWEI/
    # SIMYOU,只留在 C:/Windows/Fonts,永不拷入 fonts/)不得入训。
    # Adobe 思源双系补充字重(release 分支 raw 直链)
    for fam, repo_ in (("SourceHanSansCN", "source-han-sans"), ("SourceHanSerifCN", "source-han-serif")):
        weights = ("Light", "Heavy") if fam.endswith("Sans") else ("Regular", "Medium", "Light", "Heavy")
        for weight in weights:
            fam_tag = "Sans" if fam.endswith("Sans") else "Serif"
            cn_name = f"SourceHan{fam_tag}CN-{weight}.otf"
            t.append((cn_name,
                      lambda r=repo_, n=cn_name: (
                          f"https://github.com/adobe-fonts/{r}/raw/release/SubsetOTF/CN/{n}",
                          github_via_proxy), f"思源{'黑' if fam_tag == 'Sans' else '宋'}体 CN {weight}"))
    # 阿里普惠体补充字重(Light/Medium/Heavy,命名规律与 55/85 一致,失败可容忍)
    for code, weight in (("45", "Light"), ("65", "Medium"), ("105", "Heavy")):
        url = f"https://puhuiti.oss-cn-hangzhou.aliyuncs.com/AlibabaPuHuiTi-3/AlibabaPuHuiTi-3-{code}-{weight}/AlibabaPuHuiTi-3-{code}-{weight}.ttf"
        t.append((f"AlibabaPuHuiTi-3-{weight}.ttf", lambda u=url: (u, direct), "阿里普惠体3补充字重(国内直连)"))
    # 霞鹜文楷 GB(字形集更全的独立仓库,release 资产单文件)
    t.append(("LXGWWenKaiGB-Regular.ttf", lambda: (_lxgw_gb_url(), github_via_proxy), "霞鹜文楷 GB"))
    return t


def _lxgw_gb_url() -> str:
    data = json.loads(github_via_proxy(f"{GITHUB_API}/repos/lxgw/LxgwWenkaiGB/releases/latest"))
    for asset in data.get("assets", []):
        if asset.get("name") in ("LXGWWenKaiGB-Regular.ttf", "LXGWWenKaiGB-Medium.ttf"):
            return asset["browser_download_url"]
    raise RuntimeError("lxgw GB release 中未找到 Regular/Medium 资产")


def gh_release_assets(repo: str, want: tuple, desc: str) -> list:
    """解析 repo latest release，把资产名命中 want 的全部返回（v36 扩池通用入口）。
    zip 资产在 main() 里自动解压出内部 ttf/otf。"""
    data = json.loads(github_via_proxy(f"{GITHUB_API}/repos/{repo}/releases/latest"))
    out = []
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if any(w.lower() in name.lower() for w in want):
            out.append((name, asset["browser_download_url"]))
    if not out:
        raise RuntimeError(f"{repo} latest release 未命中 {want}")
    return [(name, url, desc) for name, url in out]


# v36 扩池源清单在 _v36_pairs()（逐源独立容错：release 解析型 + raw 直链型）


def instantiate_variable_fonts(fonts_dir: Path, ok_counter: list) -> None:
    """可变字重实例化（v36 零下载扩池）：Noto Sans/Serif SC 可变字体各出
    Light/Medium/Bold/Black 四个静态实例——同一骨架不同字重正是 App 字号差异主轴。"""
    from fontTools import varLib
    try:
        from fontTools.varLib.instancer import instantiateVariableFont
        from fontTools.ttLib import TTFont
    except ImportError:
        print("[跳过] fontTools 不可用，跳过可变字重实例化")
        return
    jobs = [("NotoSansSC-Regular.ttf", "NotoSansSC-{w}.ttf"),
            ("NotoSerifSC-Regular.ttf", "NotoSerifSC-{w}.ttf")]
    weights = (("Light", 300), ("Medium", 500), ("Bold", 700), ("Black", 900))
    for src_tpl, dst_tpl in jobs:
        src = fonts_dir / src_tpl
        if not src.exists():
            continue
        for wname, wval in weights:
            dst = fonts_dir / dst_tpl.format(w=wname)
            if dst.exists():
                ok_counter[0] += 1
                continue
            try:
                f = TTFont(str(src))
                instantiateVariableFont(f, {"wght": wval}, inplace=True)
                # 去掉可变标志，命名区分字重（family 后缀，避免同名冲突）
                for rec in f["name"].names:
                    if rec.nameID in (1, 16):
                        rec.string = f"{rec.toUnicode()} {wname}"
                f.save(str(dst))
                f.close()
                if validate_font(dst):
                    ok_counter[0] += 1
                    print(f"[实例化] {dst.name} wght={wval}")
                else:
                    dst.unlink(missing_ok=True)
                    raise RuntimeError("实例化结果无法加载")
            except Exception as e:
                print(f"[失败] {dst.name}: {e}")
                dst.unlink(missing_ok=True)


def validate_font(path: Path) -> bool:
    """用 PIL 实际加载并渲染一个汉字，验证字体文件可用（防止下载到 HTML 错误页）"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(path), 48)
        img = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(img).text((8, 8), "帅", font=font, fill=255)
        return True
    except Exception:
        return False


def sha256_head(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default=str(Path(__file__).parent / "fonts"))
    args = ap.parse_args()
    fonts_dir = Path(args.fonts_dir)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    manifest, ok, fail = [], 0, 0

    # v36 扩池：目标 = 原列表 + v36 新增源（release 解析/zip 资产自动解压）
    # v36 源统一走 github_via_proxy（release 解析型与 raw 直链型都是 GitHub 系）
    targets = [(n, t, d) for n, t, d in build_targets()] + [
        (n, (lambda u=u: (u, github_via_proxy)), d)
        for n, u, d in sum([_vt for _vt in _v36_pairs()], [])]
    for name, target, desc in targets:
        dest = fonts_dir / name
        if dest.exists() and dest.stat().st_size > 100_000 and validate_font(dest):
            print(f"[跳过] {name} 已存在且可用")
            ok += 1
            manifest.append({"file": name, "desc": desc, "bytes": dest.stat().st_size,
                             "sha256_16": sha256_head(dest.read_bytes()), "status": "cached"})
            continue
        try:
            url, fetch = target()
            print(f"[下载] {name} ← {url}")
            data = fetch(url)
            if name.lower().endswith(".zip"):
                # zip 资产（得意黑/朱雀等 release 常见）：解出首个 ttf/otf 落盘
                import io, zipfile
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.namelist()
                               if m.lower().endswith((".ttf", ".otf")) and not m.startswith("__MACOSX")]
                    if not members:
                        raise RuntimeError("zip 内无字体文件")
                    inner = Path(members[0]).name
                    dest = fonts_dir / inner
                    if dest.exists() and dest.stat().st_size > 100_000 and validate_font(dest):
                        print(f"[跳过] {inner} 已存在（zip 内已装）")
                        ok += 1
                        manifest.append({"file": inner, "desc": desc, "bytes": dest.stat().st_size,
                                         "sha256_16": sha256_head(dest.read_bytes()), "status": "cached"})
                        continue
                    data = zf.read(members[0])
                    name = inner
                    dest = fonts_dir / name
            dest.write_bytes(data)
            if not validate_font(dest):
                raise RuntimeError("下载内容无法作为字体加载")
            print(f"[完成] {name} {len(data)/1e6:.1f}MB")
            ok += 1
            manifest.append({"file": name, "desc": desc, "bytes": len(data),
                             "sha256_16": sha256_head(data), "status": "downloaded"})
        except Exception as e:
            print(f"[失败] {name}: {e}")
            fail += 1
            if dest.exists() and not dest.name.startswith("NotoSansSC"):
                pass  # 失败清理交给下方统一 unlink（可变字体源文件必须保留）
            if dest.exists() and "已存在" not in str(e):
                try:
                    if dest.name in ("NotoSansSC-Regular.ttf", "NotoSerifSC-Regular.ttf"):
                        pass  # 可变字体母本，保留供实例化
                    else:
                        dest.unlink()
                except OSError:
                    pass

    # v36：可变字重实例化（零下载扩 +8）
    okc = [0]
    instantiate_variable_fonts(fonts_dir, okc)
    ok += okc[0]

    (fonts_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    n_pool = len(list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf")))
    print(f"\n合计可用 {ok} 个字体，失败 {fail} 个；清单: {fonts_dir/'manifest.json'}")
    print(f"字体池文件数（含历史文件）: {n_pool}")
    return 0 if ok >= 10 else 1  # 方针要求 ≥10 款


def _v36_pairs():
    """build_targets_v36() 的 (name,(url,fetch),desc) 三元组展开。
    逐源独立 try：单个 repo 404/超时不连坐其他源。"""
    out = []
    # release 解析型源（每 repo 独立容错）
    import types as _t
    for repo, want, desc in (("lxgw/LxgwNeoXiHei", (".ttf", ".otf"), "霞鹜新晰黑"),
                             ("lxgw/yozai-font", ("Yozai-",), "悠哉字体"),
                             ("lxgw/smiley-sans", ("SmileySans",), "得意黑"),
                             ("TrionesType/zhuque", ("Zhuque",), "朱雀仿宋")):
        try:
            for name, url, d in gh_release_assets(repo, want, desc):
                out.append((name, url, d))
        except Exception as e:
            print(f"[失败] v36 扩池源 {repo}: {e}")
    # raw 直链型源（Noto TC/HK，不走 release API）
    for name, url, desc in (
            ("NotoSansTC-Regular.otf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Regular.otf", "思源黑体 TC"),
            ("NotoSansTC-Medium.otf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Medium.otf", "思源黑体 TC Medium"),
            ("NotoSerifTC-Regular.otf", "https://github.com/notofonts/noto-cjk/raw/main/Serif/SubsetOTF/TC/NotoSerifTC-Regular.otf", "思源宋体 TC"),
            ("NotoSansHK-Regular.otf", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/HK/NotoSansHK-Regular.otf", "思源黑体 HK")):
        out.append((name, url, desc))
    return [[(name, url, desc)] for name, url, desc in out]


if __name__ == "__main__":
    sys.exit(main())
