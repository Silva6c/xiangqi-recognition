#!/bin/bash
# 真机基准：走子 → 首次识别到变化 → board confirmed → 悬浮窗建议刷新 的时间差
#
# 用法（手机插 USB、开 USB 调试并允许授权后）：
#   bash phone/tools/bench_real_device.sh
#
# 流程：装 APK → 引导启动识别 → 开后台 logcat 监听 → 提示你走一步棋 →
# 回车停止 → 自动输出各环节精确时间差（logcat 自带时间戳，天然对齐）。
set -uo pipefail

ADB="D:/Android/Sdk/platform-tools/adb.exe"
APK="D:/AI-works/cabinet/phone/app/build/outputs/apk/debug/app-debug.apk"
PKG="com.cabinet"
OUT="${TMPDIR:-/tmp}/real_bench_logcat.txt"

DEVICE=$("$ADB" devices | awk '$2=="device" && $1 !~ /^emulator-/ {print $1; exit}')
if [ -z "$DEVICE" ]; then
    echo "[X] 未检测到真机。请插 USB 线、开启 USB 调试并允许授权弹窗后重试。"
    echo "    当前设备列表："
    "$ADB" devices
    exit 1
fi
echo "[1/5] 真机已连接: $DEVICE"

"$ADB" -s "$DEVICE" install -r "$APK" | tail -1 || exit 1
echo "[2/5] 最新 APK 已安装（量化分类模型 + 2 帧确认 + burst 快速确认）"

# 悬浮窗权限免手动设置
"$ADB" -s "$DEVICE" shell appops set "$PKG" SYSTEM_ALERT_WINDOW allow 2>/dev/null
"$ADB" -s "$DEVICE" shell "logcat -c"

echo "[3/5] 请在手机上完成（此时监听尚未开始）："
echo "      a) 打开 内阁 → 点『开始屏幕识别』→ 系统弹窗『立即开始』"
echo "      b) 切到 cnvcs 对弈界面，等悬浮窗出现第一条建议（基线）"
echo "      c) 按回车 → 脚本开始监听 → 在棋盘上走一步棋"
read -r -p "基线建议已出现,按回车开始监听: " _

: > "$OUT"
"$ADB" -s "$DEVICE" logcat -v time > "$OUT" 2>&1 &
LOGPID=$!
echo "      >>> 现在在棋盘上走一步棋,走完等约 5 秒 <<<"
read -r -p "走完棋后按回车结束监听: " _
kill "$LOGPID" 2>/dev/null; wait "$LOGPID" 2>/dev/null

echo ""
echo "[4/5] 各环节时间线（logcat 时间戳，天然对齐）:"
grep -E "rec\(onnx\): pieces|board confirmed|position sent|suggestion received" "$OUT" | tail -8

echo ""
FIRST_FEN=$(grep -oE "[^ ]+ [^ ]+ .*rec\(onnx\)" "$OUT" | tail -2 | head -1 | awk '{print $1, $2}')
CONFIRMED=$(grep "board confirmed" "$OUT" | tail -1 | awk '{print $1, $2}')
SENT=$(grep "position sent" "$OUT" | tail -1 | awk '{print $1, $2}')
SUGGEST=$(grep "suggestion received" "$OUT" | tail -1 | awk '{print $1, $2}')

echo "[5/5] 时间差汇总（全在手机 logcat 时间轴上）:"
python - "$FIRST_FEN" "$CONFIRMED" "$SENT" "$SUGGEST" <<'PY'
import sys, datetime
def parse(s):
    try:
        return datetime.datetime.strptime(s, "%m-%d %H:%M:%S.%f")
    except Exception:
        return None
first, conf, sent, sug = (parse(x) for x in sys.argv[1:5])
def diff(a, b, label):
    if a and b:
        print(f"  {label}: {(b - a).total_seconds():.2f} 秒")
    else:
        print(f"  {label}: 缺少时间点（抓取窗口太短?）")
diff(first, conf, "首次识别到新局面 → 确认(2帧)")
diff(conf, sent, "确认 → 转发完成(发给 PC)")
diff(sent, sug,  "转发 → 悬浮窗建议刷新(含PC引擎)")
diff(first, sug, "合计: 识别管线+转发+PC引擎")
PY
echo ""
echo "注: PC 端 logs/cabinet.log 的『收到局面』可交叉验证（两端时钟可能有秒级偏差）。"
