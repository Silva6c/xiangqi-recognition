package com.cabinet.assist

import com.cabinet.gamelogic.Piece
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * DetectionBoardMapper 朝向判定单测（纯 JVM）。
 *
 * 重点回归（2026-10-01 模拟器实测）：RTMPose 角点身份整体漂移会让 warp 把棋盘
 * 采成旋转 180° 的俯视图，双王内容判定（detectOrientation）此时反映的是 warp
 * 内部朝向而非屏幕朝向；确认层曾把该错误朝向锁存，"我方执色/小地图/轮次"随之
 * 永久颠倒。screenOrientation 用角点在原图的几何位置+内容判定联合推断屏幕朝向，
 * 无论 warp 是否旋转、无论 pose 用何种角点语义，结果恒正确且逐帧无状态。
 */
class DetectionBoardMapperTest {

    /** 正立 warp 的角点：kp0=左上 kp1=右上 kp2=左下 kp3=右下（画面即棋盘正放） */
    private val upright = floatArrayOf(
        100f, 100f, 900f, 100f, 100f, 1000f, 900f, 1000f)

    /**
     * 旋转 180° 的 warp 角点（模拟器事故场景）：pose 把 kp0 输出在屏幕右下、
     * kp1 左下、kp2 右上、kp3 左上——warp 顶边落在屏幕下方。
     */
    private val rotated = floatArrayOf(
        900f, 1000f, 100f, 1000f, 900f, 100f, 100f, 100f)

    private fun rotate180(src: Array<IntArray>): Array<IntArray> {
        val out = Array(AssistBoard.H) { IntArray(AssistBoard.W) }
        for (y in 0 until AssistBoard.H) for (x in 0 until AssistBoard.W) {
            out[AssistBoard.H - 1 - y][AssistBoard.W - 1 - x] = src[y][x]
        }
        return out
    }

    @Test
    fun `标准画面正立warp判定STANDARD`() {
        assertEquals(Orientation.STANDARD,
            DetectionBoardMapper.screenOrientation(upright, Orientation.STANDARD))
    }

    @Test
    fun `标准画面旋转warp仍判定STANDARD_模拟器事故场景`() {
        // warp 旋转 → raw 是旋转布局（红在 raw 第 0 行）→ 内容判定报 FLIPPED；
        // 几何判定必须纠回屏幕真实朝向 STANDARD
        assertEquals(Orientation.FLIPPED,
            DetectionBoardMapper.detectOrientation(rotate180(AssistBoard.canonicalStart())))
        assertEquals(Orientation.STANDARD,
            DetectionBoardMapper.screenOrientation(rotated, Orientation.FLIPPED))
    }

    @Test
    fun `翻转画面正立warp判定FLIPPED`() {
        assertEquals(Orientation.FLIPPED,
            DetectionBoardMapper.screenOrientation(upright, Orientation.FLIPPED))
    }

    @Test
    fun `翻转画面旋转warp仍判定FLIPPED`() {
        // 画面翻转（红上）+ warp 旋转 → raw 恢复标准布局（红在 raw 第 9 行）→
        // 内容判定报 STANDARD；几何判定纠回屏幕真实朝向 FLIPPED
        assertEquals(Orientation.STANDARD,
            DetectionBoardMapper.detectOrientation(AssistBoard.canonicalStart()))
        assertEquals(Orientation.FLIPPED,
            DetectionBoardMapper.screenOrientation(rotated, Orientation.STANDARD))
    }

    @Test
    fun `detectOrientation双王行位判定标准与翻转`() {
        assertEquals(Orientation.STANDARD,
            DetectionBoardMapper.detectOrientation(AssistBoard.canonicalStart()))
        assertEquals(Orientation.FLIPPED,
            DetectionBoardMapper.detectOrientation(rotate180(AssistBoard.canonicalStart())))
        // 双王皆缺：返回 null（调用方按"未识别"丢帧）
        val noKings = Array(AssistBoard.H) { IntArray(AssistBoard.W) }
        assertTrue(DetectionBoardMapper.detectOrientation(noKings) == null)
    }

    @Test
    fun `toCanonical标准原样_翻转旋转180恢复`() {
        val start = AssistBoard.canonicalStart()
        assertTrue(AssistBoard.equal(start,
            DetectionBoardMapper.toCanonical(start, Orientation.STANDARD)))
        assertTrue(AssistBoard.equal(start,
            DetectionBoardMapper.toCanonical(rotate180(start), Orientation.FLIPPED)))
    }

    @Test
    fun `旋转warp整链自洽_内容自愈canonical加几何纠回屏幕朝向`() {
        // 端到端语义回归：warp 旋转帧 → canonical 恒为规范布局（红在 y=9）、
        // orientation 恒为真实屏幕朝向（STANDARD 画面），二者互不污染
        val raw = rotate180(AssistBoard.canonicalStart())
        val content = DetectionBoardMapper.detectOrientation(raw)!!
        val canonical = DetectionBoardMapper.toCanonical(raw, content)
        val screen = DetectionBoardMapper.screenOrientation(rotated, content)
        assertTrue(AssistBoard.equal(AssistBoard.canonicalStart(), canonical))
        assertEquals(Orientation.STANDARD, screen)
        // 规范布局首行必为黑方底线、末行必为红方底线
        assertEquals(Piece.BJIANG, canonical[0][4])
        assertEquals(Piece.WSHUAI, canonical[9][4])
    }
}
