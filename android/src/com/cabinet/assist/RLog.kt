package com.cabinet.assist

import android.util.Log
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 可上行日志：本应用诊断日志的统一出口——照常写进系统 logcat（tag 不变，
 * adb logcat 排查习惯不受影响），同时进入内存队列，由 [RemoteAnalysisClient]
 * 定时批量取走（[drain]）经 WSS 上行，PC 端落盘到 logs/phone/ 指定文件夹
 * （2026-09-30 用户提议：真机测试发现问题无需连线抓 logcat）。
 *
 * 队列为环形（超过 [MAX_QUEUE] 丢最旧）：连接断开期间日志不会无限积压，
 * 诊断辅助的定位就是"尽力而为"，绝不挤占对局链路。线程安全（WSS 线程与
 * 各识别线程并发写入）。
 *
 * ⚠【雷区】环形队列意味着 debug 噪音会把关键日志挤出去——热路径（每帧/
 * 每次识别）别加高频 d() 日志；需要留排查线索时优先 i() 且消息里带上关键
 * 状态量（子数/FEN/事件），一行顶十行。每条日志都过 SimpleDateFormat 同步
 * format，热路径无条件拼大字符串本身就是开销（2026-10-02 性能轮实测）。
 */
object RLog {
    /** 队列上限（行）：按每步棋约 10 条估算，断连 40s+ 的量也不会丢关键段 */
    private const val MAX_QUEUE = 400

    private val queue = ArrayDeque<String>()
    /** 因队列满被丢弃的行数累计（发送方经 [takeDropped] 取走补报，让 PC 端知道有缺口） */
    private var droppedCount = 0L

    // SimpleDateFormat 非线程安全，各线程 format 时同步
    private val tsFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    fun d(tag: String, msg: String) = log(Log.DEBUG, tag, msg)
    fun i(tag: String, msg: String) = log(Log.INFO, tag, msg)
    fun w(tag: String, msg: String, tr: Throwable? = null) = log(Log.WARN, tag, msg, tr)
    fun e(tag: String, msg: String, tr: Throwable? = null) = log(Log.ERROR, tag, msg, tr)

    /** 取走当前积攒的全部日志行（发送方定时调用） */
    fun drain(): List<String> {
        synchronized(queue) {
            val out = queue.toList()
            queue.clear()
            return out
        }
    }

    /** 取走累计丢弃行数并清零（发送方随下一批日志补报一行说明） */
    fun takeDropped(): Long {
        synchronized(queue) {
            val d = droppedCount
            droppedCount = 0
            return d
        }
    }

    private fun log(priority: Int, tag: String, msg: String, tr: Throwable? = null) {
        // logcat 照常：线上排查与 AS Logcat 窗口不受影响
        if (tr != null) Log.println(priority, tag, "$msg\n${Log.getStackTraceString(tr)}")
        else Log.println(priority, tag, msg)
        val level = when (priority) {
            Log.VERBOSE -> "V"; Log.DEBUG -> "D"; Log.INFO -> "I"
            Log.WARN -> "W"; else -> "E"
        }
        val full = if (tr != null) "$msg | ${tr.javaClass.simpleName}: ${tr.message}" else msg
        val line = "${tsFormat.format(Date())} $level/$tag: $full"
        synchronized(queue) {
            if (queue.size >= MAX_QUEUE) {
                queue.removeFirst()
                droppedCount++
            }
            queue.addLast(line)
        }
    }
}
