package com.cabinet.assist

import android.content.res.AssetManager
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * v63 两级引擎第二级：cell_cls 格级精分类器 + 级联仲裁（纯自研，零 nano）。
 *
 * 与离线评测器 phone/tools/recog/eval_cascade_cell.py 终版逐条对齐
 * （26 帧 v2：r3 单模型 11 → 两级引擎 2 追平 nano，docs/v63方案）：
 *  - gate 采纳：r3 softmax conf < 0.90 的格采纳 cell argmax（cell conf ≥ 0.60 双门）
 *  - conflict 复核：r3 ≥ 0.92 且 cell ≥ 0.95 且意见相左 → 采纳 cell
 *    （0.97→0.92：bench 三帧 (9,7) R→N 老顽固格 cell 全对 0.993 却被 0.97 误拦，
 *    放宽后 bench 10/10 收复且 26 帧 LO=2/HD=8 零伤——docs/v63方案 §七）
 *  - 帅将双保护：r3 判 K/k 的格不覆盖、不参与子数翻格（r3 全盘史无 K/k 错型，
 *    cell 的 K/k 判别是软肋——26 帧实测两误伤皆此型）
 *  - 子数四档仲裁：占子 ≥8 才跑（空盘/残局不跑——gold49 低质帧连锁翻格教训）；
 *    候选优先级 cell 指认(≥0.60) > r3 次选缺编 > cell 空格 > r3 次选空格
 *  - 改格数闸：级联改动 > 13 格（15%）= 帧级乱套信号，整帧弃用级联保原判
 *    （正常帧级联只改 1~5 格；低质帧防线，bad_frame 重探由上层既有机制承接）
 *
 * cell 输入 = warp450 帧格心 ±20px 的 40×40 crop → 128×128（LO 口径——与离线
 * 评测最优口径完全一致）；预处理 /255（注意：与主分类器的 ImageNet mean/std 不同，
 * cell_cls 训练即 /255）。fp32 直接部署（153K 小模型实测 int8 反慢且掉点，弃用）。
 */
class CellCascade private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
) {
    companion object {
        const val MODEL_PATH = "models/cell_cls_fp32.onnx"
        private const val CELL_IN = 128
        private const val NCLASS = 16
        private const val NCELLS = 90

        /** warp450 空间格心（与 OnnxGeom/renderer 契约同源）：网格角 (50,50)-(400,450) */
        private fun cellCx(c: Int) = 50.0 + 43.75 * c
        private fun cellCy(r: Int) = 50.0 + 44.4444 * r

        // ---- 级联参数（eval_cascade_cell.py 终版口径，docs/v63方案 §三）----
        private const val THR_GATE = 0.90      // r3 低置信门
        private const val THR_CELL = 0.60      // cell 双门
        private const val THR_R3_HI = 0.92     // conflict 的 r3 中高置信门（0.97→0.92 bench 收复定案）
        private const val THR_CELL_HI = 0.95   // conflict 的 cell 高置信门
        private const val MAX_CHANGED = 13     // 改格数闸（90 格的 15%）

        /** 每类上限（16 类索引：2K 3A 4B 5N 6R 7C 8P / 9k 10a 11b 12n 13r 14c 15p） */
        private val LIMITS = intArrayOf(0, 0, 1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5)

        /** 加载失败返回 null（assets 缺模型 → 调用方回退纯单级，与 v45 缺文件回退同款） */
        fun load(assets: AssetManager): CellCascade? = try {
            val env = OrtEnvironment.getEnvironment()
            val bytes = assets.open(MODEL_PATH).use { it.readBytes() }
            val so = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            val session = env.createSession(bytes, so)
            val input = session.inputNames.iterator().next()
            CellCascade(env, session, input)
        } catch (t: Throwable) {
            RLog.w(TAG, "cell_cls 加载失败（级联不启用）: ${t.message}")
            null
        }

        private const val TAG = "CellCascade"

        /**
         * 纯决策逻辑（无推理依赖，JVM 单测直接测）：给定主分类器 softmax 后的
         * top1/second/pmax 与 cell_cls 的 cellTop/cellConf（未触发的格 cellTop=-1），
         * 输出修正后的 90 格类索引与统计。四通道+双保护+子数+改格数闸全在此。
         */
        fun decide(top1: IntArray, second: IntArray, pmax: FloatArray,
                   cellTop: IntArray, cellConf: FloatArray): Result {
            val pred = top1.copyOf()
            var changed = 0
            for (g in 0 until NCELLS) {
                val cp = cellTop[g]
                if (cp < 0 || cp == 1 || cp == pred[g]) continue  // 未触发/x 伪类/一致不动（空格 0 可采纳=幻视修复通道）
                if (pmax[g] < THR_GATE && cellConf[g] >= THR_CELL && pred[g] !in intArrayOf(2, 9)) {
                    pred[g] = cp; changed++
                } else if (pmax[g] >= THR_R3_HI && cellConf[g] >= THR_CELL_HI
                    && pred[g] !in intArrayOf(2, 9) && cp !in intArrayOf(2, 9)) {
                    pred[g] = cp; changed++
                }
            }
            var occupied = 0
            for (g in 0 until NCELLS) if (pred[g] != 0) occupied++
            if (occupied >= 8) {
                for (round in 0 until 10) {
                    val counts = IntArray(NCLASS)
                    for (g in 0 until NCELLS) counts[pred[g]]++
                    var progressed = false
                    for (cls in 2 until NCLASS) {
                        if (cls == 9 || counts[cls] <= LIMITS[cls]) continue
                        val deficit = (2 until NCLASS).filter { it != 9 && counts[it] < LIMITS[it] }
                        var best: IntArray? = null
                        for (g in 0 until NCELLS) {
                            if (pred[g] != cls) continue
                            val s2 = second[g]; val cp = cellTop[g]
                            val cand = when {
                                cp in deficit && cellConf[g] >= THR_CELL -> intArrayOf(0, (pmax[g] * 1000).toInt(), g, cp)
                                s2 in deficit -> intArrayOf(1, (pmax[g] * 1000).toInt(), g, s2)
                                cp == 0 && cellConf[g] >= THR_CELL -> intArrayOf(2, (pmax[g] * 1000).toInt(), g, 0)
                                s2 == 0 -> intArrayOf(3, (pmax[g] * 1000).toInt(), g, 0)
                                else -> null
                            }
                            if (cand != null && (best == null || cand[0] < best!![0] ||
                                        (cand[0] == best!![0] && cand[1] < best!![1]))) best = cand
                        }
                        if (best != null) { pred[best[2]] = best[3]; changed++; progressed = true }
                    }
                    if (!progressed) break
                }
            }
            if (changed > MAX_CHANGED) {
                // 决策函数保持 JVM 纯净（单测可跑）；unstable 日志由调用方记录
                return Result(top1, 0, true)
            }
            return Result(pred, changed, false)
        }
    }

    fun close() {
        try { session.close() } catch (_: Throwable) {}
    }

    // ---- 复用缓冲（arbitrate 由 recognize 的 @Synchronized 串行保护，同主识别器雷区约定）----
    private val cropBuf = IntArray(40 * 40)
    private val resizeBuf = IntArray(CELL_IN * CELL_IN)
    private val inputArr = FloatArray(16 * 3 * CELL_IN * CELL_IN)   // 16 格/批（分批上限）
    private val inputBuf: FloatBuffer = ByteBuffer.allocateDirect(inputArr.size * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()

    /** 级联结果：修正后的 90 格类索引（0-15）与统计。unstable=true 时 changed 全还原。 */
    class Result(val cells: IntArray, val changed: Int, val unstable: Boolean)

    /**
     * 级联仲裁入口。
     * @param logits 主分类器 90×16 logits（decodeCells 前的原始输出）
     * @param warp450 450×500 ARGB 帧与 OnnxBoardRecognizer.boardWarpOut 同源
     * @param warpW warp 帧宽（450）
     */
    fun arbitrate(logits: FloatArray, warp450: IntArray, warpW: Int): Result {
        // ---- softmax → top1/pmax/second（主分类器 decodeCells 只取 argmax，conf 在此补算）----
        val top1 = IntArray(NCELLS); val second = IntArray(NCELLS)
        val pmax = FloatArray(NCELLS)
        val probs = FloatArray(NCLASS)
        for (i in 0 until NCELLS) {
            var b0 = 0; var b1 = 1; var v0 = logits[i * NCLASS]; var v1 = Float.NEGATIVE_INFINITY
            var mx = v0
            for (j in 1 until NCLASS) {
                val v = logits[i * NCLASS + j]
                if (v > mx) mx = v
                if (v > v0) { v1 = v0; b1 = b0; v0 = v; b0 = j } else if (v > v1) { v1 = v; b1 = j }
            }
            var sum = 0f
            for (j in 0 until NCLASS) { val e = kotlin.math.exp(logits[i * NCLASS + j] - mx); probs[j] = e; sum += e }
            top1[i] = b0; second[i] = b1; pmax[i] = probs[b0] / sum
        }
        val pred = top1.copyOf()

        // ---- 触发格收集（gate + conflict 并集）----
        val want = BooleanArray(NCELLS)
        for (i in 0 until NCELLS) {
            val gate = pmax[i] < THR_GATE && pred[i] !in intArrayOf(2, 9)
            val conflict = pmax[i] >= THR_R3_HI
            if (gate || conflict) want[i] = true
        }
        val idx = (0 until NCELLS).filter { want[it] }
        if (idx.isEmpty()) return Result(pred, 0, false)

        // ---- cell_cls 批量推理（LO crop：warp450 格心 ±20px；16 格/批控制临时内存）----
        val cellTop = IntArray(NCELLS) { -1 }
        val cellConf = FloatArray(NCELLS)
        var k0 = 0
        while (k0 < idx.size) {
            val nb = minOf(16, idx.size - k0)
            var off = 0
            for (t in 0 until nb) {
                val g = idx[k0 + t]
                cropCell(warp450, warpW, g, cropBuf)
                OnnxGeom.resizeBilinearInto(cropBuf, 40, 40, CELL_IN, CELL_IN, resizeBuf)
                normalizeInto(resizeBuf, inputArr, off)
                off += 3 * CELL_IN * CELL_IN
            }
            java.util.Arrays.fill(inputArr, off, inputArr.size, 0f)  // 不足 16 格补零：
            // direct buffer 是"引用内存"，容量必须与 tensor shape 精确相等（ORT 契约），
            // 动态分配每批新建 400KB 级 buffer 不划算，固定批 16 换 buffer 恒定
            val logitsCell = inferBatch(inputArr.size / (3 * CELL_IN * CELL_IN))
            for (t in 0 until nb) {
                val g = idx[k0 + t]
                val base = t * NCLASS
                var b = 0; var mx = logitsCell[base]
                var sum = 0f
                val exps = FloatArray(NCLASS)
                for (j in 1 until NCLASS) if (logitsCell[base + j] > mx) { mx = logitsCell[base + j]; b = j }
                for (j in 0 until NCLASS) { val e = kotlin.math.exp(logitsCell[base + j] - mx); exps[j] = e; sum += e }
                cellTop[g] = b; cellConf[g] = exps[b] / sum
            }
            k0 += nb
        }

        // ---- 纯决策（gate/conflict/子数/闸，companion.decide 与离线评测器对齐）----
        return decide(top1, second, pmax, cellTop, cellConf)
    }

    /** warp450 帧上抠格 g（r=g/9, c=g%9）的 40×40 crop（格心 ±20px，边界钳制）。 */
    private fun cropCell(warp: IntArray, warpW: Int, g: Int, out: IntArray) {
        val r = g / 9; val c = g % 9
        val cx = cellCx(c).toInt(); val cy = cellCy(r).toInt()
        val x0 = (cx - 20).coerceAtLeast(0); val y0 = (cy - 20).coerceAtLeast(0)
        val x1 = (cx + 20).coerceAtMost(warpW); val y1 = (cy + 20).coerceAtMost(500)
        var o = 0
        for (y in y0 until y1) {
            System.arraycopy(warp, y * warpW + x0, out, o, x1 - x0)
            o += x1 - x0
        }
        // 边界钳制产生的短行/短列以 0 填充余量（40×40 定长，resize 按整块读）
        while (o < out.size) out[o++] = 0
    }

    /** ARGB IntArray → /255 CHW float（cell_cls 训练口径：无 mean/std）。 */
    private fun normalizeInto(argb: IntArray, dst: FloatArray, off: Int) {
        val n = argb.size
        for (i in 0 until n) {
            val p = argb[i]
            dst[off + i] = ((p shr 16 and 0xFF)) / 255f
            dst[off + n + i] = ((p shr 8 and 0xFF)) / 255f
            dst[off + 2 * n + i] = ((p and 0xFF)) / 255f
        }
    }

    private fun inferBatch(batch: Int): FloatArray {
        inputBuf.rewind(); inputBuf.put(inputArr); inputBuf.rewind()
        val shape = longArrayOf(batch.toLong(), 3, CELL_IN.toLong(), CELL_IN.toLong())
        OnnxTensor.createTensor(env, inputBuf, shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { res ->
                val out = res[0].value as Array<FloatArray>
                val flat = FloatArray(batch * NCLASS)
                for (i in 0 until batch) System.arraycopy(out[i], 0, flat, i * NCLASS, NCLASS)
                return flat
            }
        }
    }
}
