package com.cabinet.assist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * v63 两级引擎决策逻辑单测（CellCascade.decide 纯函数，与离线评测器
 * eval_cascade_cell.py 终版对齐）。类索引：0'.' 2K 5N 6R 8P 9k 12n 13r 15p。
 */
class CellCascadeDecideTest {

    private fun blank(): Pair<IntArray, FloatArray> =
        IntArray(90) { -1 } to FloatArray(90)

    private fun baseBoard(top1: IntArray, second: IntArray = IntArray(90), pmax: FloatArray): Triple<IntArray, IntArray, FloatArray> =
        Triple(top1.copyOf(), second.copyOf(), pmax.copyOf())

    /** 1. gate 采纳：r3 低置信错 + cell 高置信对 → 翻正 */
    @Test
    fun gateAdopt() {
        val top1 = IntArray(90) { 0 }                       // 全空格
        top1[10] = 15                                        // p（错，真值空）
        val pmax = FloatArray(90) { 0.99f }; pmax[10] = 0.5f
        val second = IntArray(90)
        val (ct, cc) = blank(); ct[10] = 0; cc[10] = 0.9f    // cell 判空格
        val r = CellCascade.decide(top1, second, pmax, ct, cc)
        assertEquals(0, r.cells[10])
        assertEquals(1, r.changed)
        assertFalse(r.unstable)
    }

    /** 2. 帅将保护：r3 判 K（无论置信多低）不被 cell 覆盖 */
    @Test
    fun kingProtectionGate() {
        val top1 = IntArray(90) { 0 }; top1[5] = 2           // K
        val pmax = FloatArray(90) { 0.99f }; pmax[5] = 0.3f
        val (ct, cc) = blank(); ct[5] = 0; cc[5] = 0.99f
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertEquals(2, r.cells[5])                          // 保持 K
    }

    /** 3. conflict 复核：r3 0.99 高置信 vs cell 0.97 高置信分歧 → 采纳 cell */
    @Test
    fun conflictAdopt() {
        val top1 = IntArray(90) { 0 }; top1[7] = 5           // N（r3 说 N）
        val pmax = FloatArray(90) { 0.99f }
        val (ct, cc) = blank(); ct[7] = 6; cc[7] = 0.97f     // cell 说 R
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertEquals(6, r.cells[7])
    }

    /** 3b. conflict 帅将双向保护：cell 预测为 K/k 时即便双高置信也不采纳 */
    @Test
    fun conflictKingBlocked() {
        val top1 = IntArray(90) { 0 }; top1[7] = 5
        val pmax = FloatArray(90) { 0.99f }
        val (ct, cc) = blank(); ct[7] = 2; cc[7] = 0.99f     // cell 说 K
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertEquals(5, r.cells[7])
    }

    /** 3c. conflict 新门（0.92）：r3 0.964 中高置信 vs cell 0.993 → 采纳 cell
     *  （bench#7 (9,7) R→N 型：旧门 0.97 误拦的格，0.92 后翻正） */
    @Test
    fun conflictMidConfAdopt() {
        val top1 = IntArray(90) { 0 }; top1[7] = 5           // N（r3 说 N@0.964）
        val pmax = FloatArray(90) { 0.99f }; pmax[7] = 0.964f
        val (ct, cc) = blank(); ct[7] = 6; cc[7] = 0.993f    // cell 说 R@0.993
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertEquals(6, r.cells[7])
    }

    /** 4. cthr 双门：cell conf 不足（0.5<0.6）不采纳 */
    @Test
    fun cellConfGate() {
        val top1 = IntArray(90) { 0 }; top1[10] = 15
        val pmax = FloatArray(90) { 0.99f }; pmax[10] = 0.5f
        val (ct, cc) = blank(); ct[10] = 0; cc[10] = 0.5f
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertEquals(15, r.cells[10])
    }

    /** 5. 子数 cell 指认：黑马 3 个（超编）+ 红炮缺编 + cell 指认红炮 → 翻 */
    @Test
    fun ruleCellAccuse() {
        val top1 = IntArray(90) { 0 }
        top1[0] = 2; top1[80] = 9                            // 帅将各一（占子凑 8）
        for (i in 1..3) top1[i] = 12                          // 黑马 ×3（超编）
        top1[10] = 6; top1[11] = 6                            // 红车 ×2
        top1[20] = 8; top1[21] = 8                            // 红兵 ×2
        val pmax = FloatArray(90) { 0.99f }
        val (ct, cc) = blank(); ct[3] = 7; cc[3] = 0.9f       // cell 指认红炮 C=7
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertEquals(7, r.cells[3])                          // 超编马 → cell 指认的炮
        var n12 = 0; for (g in 0 until 90) if (r.cells[g] == 12) n12++
        assertEquals(2, n12)                                 // 马回到上限 2
    }

    /** 6. 改格数闸：>13 格改动 → unstable 整帧还原 */
    @Test
    fun unstableGuard() {
        val top1 = IntArray(90) { 0 }
        for (i in 0 until 20) top1[i] = 15                    // 20 个卒
        val pmax = FloatArray(90) { 0.5f }                     // 全低置信
        val (ct, cc) = blank()
        for (i in 0 until 20) { ct[i] = 0; cc[i] = 0.9f }      // cell 全说空格
        val r = CellCascade.decide(top1, IntArray(90), pmax, ct, cc)
        assertTrue(r.unstable)
        assertEquals(top1, r.cells)                           // 还原原判
    }

    /** 7. 空盘保护：占子 <8 不跑子数（残局幻视不连锁翻格） */
    @Test
    fun sparseBoardNoRule() {
        val top1 = IntArray(90) { 0 }
        top1[0] = 12; top1[1] = 12; top1[2] = 12              // 3 马（超编但占子仅 3）
        val pmax = FloatArray(90) { 0.99f }
        val second = IntArray(90)
        val (ct, cc) = blank()
        val r = CellCascade.decide(top1, second, pmax, ct, cc)
        var n12 = 0; for (g in 0 until 90) if (r.cells[g] == 12) n12++
        assertEquals(3, n12)                                 // 未动
    }
}
