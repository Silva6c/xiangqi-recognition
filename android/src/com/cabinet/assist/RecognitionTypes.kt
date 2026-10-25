package com.cabinet.assist

import kotlin.math.abs

/**
 * 识别管线的共享类型（纯 JVM，无 Android 依赖）：
 * 帧输入、朝向、单帧识别结果、稳定帧跟踪器。
 */
class Frame(val width: Int, val height: Int, val argb: IntArray) {
    init {
        // 像素数必须严格等于 w*h：Frame 由 ImageReader 逐行拷贝组装，一旦行跨度/
        // 尺寸传错，识别层只会看到"整体错位"的画面而无声出错——这里 fail-fast，
        // 宁可当帧抛异常也不让坏帧流进管线（warp/letterbox 均假定像素紧凑排列）
        require(argb.size == width * height) { "frame pixel count mismatch" }
    }
}

enum class Orientation {
    /** 红方在下（标准朝向，屏幕下方阵营=红方） */
    STANDARD,
    /** 红方在上（换边/翻转朝向，屏幕下方阵营=黑方） */
    FLIPPED;
}

/**
 * 一帧识别结果。canonical 为项目内部约定（红恒定在 y=9），orientation 表示屏幕朝向。
 */
class RecognitionResult(
    val canonical: Array<IntArray>,
    val orientation: Orientation,
    val issues: List<String>,
    val recognizedPieces: Int,
) {
    fun isValid(): Boolean = issues.isEmpty()
}

/**
 * 稳定帧跟踪器：连续 confirmCount 帧出现同一局面才确认，并推断轮次/检测重开。
 * @param singleMoveRepair v39 走子连贯性开关（docs/走子连贯性设计与App域解剖-20261017.md）：
 *   确认时刻做"单步走子匹配"——枚举旧局面的全部伪合法走子，d==0 直接锁定走子方，
 *   恰好差 1 格且唯一走子可解释则修正后确认（漏检主形态"马跳走原地残留"被规则修复）；
 *   d≥2 或歧义维持现状，12 帧自愈继续兜底。默认开。
 */
class BoardTracker(private val confirmCount: Int = 3, private val singleMoveRepair: Boolean = true) {

    enum class Event {
        /** 稳定新局确认 */
        NEW_BOARD,
        /** 与当前确认局面相同（未变化） */
        SAME_BOARD,
        /** 检测到新对局（回到开局） */
        NEW_GAME,
        /** 帧不稳定/不合法，等待后续 */
        UNSTABLE,
    }

    /** 当前确认局面 */
    var confirmed: RecognitionResult? = null
        private set

    /** 红方走子方（true=红方走）；worker 写、主线程读（悔棋/更新棋局按钮） */
    @Volatile var redGo = true

    /** 连续未确认帧数（用于提示）；worker 写、主线程读 */
    @Volatile var unstableStreak = 0
        private set

    /** 单步匹配修正累计生效次数（d==1 修正 + d==0 锁定轮次；诊断/验证用） */
    @Volatile var repairCount = 0
        private set

    /**
     * 候选累计中：已识别到与确认局面不同的新局面、但尚未满 confirmCount 连续帧。
     * frameTick 据此重放补帧——点选式走子（无动画帧）时画面变化标志会被单次识别
     * 消费，若漏判此状态且画面静止，确认链将永久饿死（实测：对弈 App 即时落子后
     * 识别停在旧局面、两次识别恰好夹着对方回手，各只累计 1 帧）。
     * worker 写、主线程读。
     */
    @Volatile var confirming: Boolean = false
        private set

    private var candidate: RecognitionResult? = null
    private var candidateHits = 0

    // ---- B1 轮次恢复素材（2026-10-24）：进程重启前最后上行的局面 ----
    // 纯盘面 + 轮次，首次确认消费后即清；不持久化在 tracker 内（由服务层从
    // AssistConfig 注入，tracker 保持纯 JVM 无 Android 依赖）。
    private var restoreBoard: Array<IntArray>? = null
    private var restoreRedGo = true

    /**
     * 注入持久化恢复素材（进程启动时调用一次）：中局面首次确认时用于恢复走子方
     * ——重启后 redGo 恒 true，稳态局面 PC 校验拦不住，错轮次会让建议被端上
     * myTurn 门禁全部压掉（"重启后永远没建议"）。FEN 解析失败静默放弃。
     */
    fun setRestoreHint(fen: String) {
        val board = AssistBoard.fromFenBoard(fen) ?: return
        val parts = fen.split(" ")
        restoreBoard = board
        restoreRedGo = parts.size > 1 && parts[1] == "w"
    }

    fun reset(redGoFirst: Boolean = true) {
        confirmed = null
        candidate = null
        candidateHits = 0
        unstableStreak = 0
        confirming = false
        redGo = redGoFirst
    }

    /** 悔棋：恢复到历史局面（以给定结果与轮次重建跟踪状态，后续帧可正常重新确认） */
    fun restore(result: RecognitionResult, redGoSide: Boolean) {
        confirmed = result
        redGo = redGoSide
        candidate = null
        candidateHits = 0
        unstableStreak = 0
        confirming = false
    }

    fun onFrame(res: RecognitionResult): Event {
        val e = onFrameInner(res)
        confirming = candidate != null && candidateHits < confirmCount
        return e
    }

    private fun onFrameInner(res: RecognitionResult): Event {
        // 长期不稳定自愈：已确认局面过时后（换局/识别漂移/幽灵局面被误确认），
        // 子数跳变门限会把真实新局面永久拒之门外——每帧 candidate 被清、UNSTABLE
        // 无限持续、建议被 recHealthy 抑制，用户看到的就是"卡住不动"。
        // 连续 12 帧（慢路径约 15 秒）未确认即丢弃旧确认，重新走首次确认流程
        // （仍需连续 confirmCount 帧一致，瞬时坏帧/幽灵帧难以蒙混）。
        if (unstableStreak >= 12) {
            confirmed = null
            candidate = null
            candidateHits = 0
            unstableStreak = 0 // 同步清零:只复位一次,否则下一帧再次清掉正在累计的候选
        }
        val prev = confirmed
        // 坏帧不清候选：isValid=false 直接返回，candidate/candidateHits 原样保留——
        // "连续 confirmCount 帧"只对有效帧计数，瞬时遮挡/动画坏帧不打断好帧的累计；
        // 只有"有效但内容不同"的帧才会重置候选。若在此清候选，偶发一帧坏帧就会
        // 让走子确认永远差一口气。
        if (!res.isValid()) {
            unstableStreak++
            return Event.UNSTABLE
        }

        if (prev == null) {
            // 首次确认：连续 confirmCount 帧完全一致才确认（瞬时坏帧/幻觉帧很难
            // 连续多帧逐格一致）；确认时与开局完全一致才视为全新对局
            if (candidate != null && AssistBoard.equal(res.canonical, candidate!!.canonical)) {
                candidateHits++
            } else {
                candidate = res
                candidateHits = 1
            }
            if (candidateHits >= confirmCount) {
                confirmed = candidate!!
                val first = candidate!!
                candidate = null
                candidateHits = 0
                unstableStreak = 0
                // 容错：(9,7) 顽固错位与吃子局面的区分见 AssistBoard.isStartLike
                val isNewGame = AssistBoard.isStartLike(first.canonical)
                if (isNewGame) {
                    // 与下方"重开检测"路径对齐：开局=新对局红先。此路径多发生在
                    // 12 帧自愈之后，redGo 若保留上一局残局的旧值，新局会以错误
                    // 轮次上行（2026-10-01 实测链路的一环）
                    redGo = true
                    restoreBoard = null // 开局天然红先，恢复素材作废
                } else {
                    // B1 轮次恢复（2026-10-24）：中局面首次确认时按持久化素材恢复
                    // 走子方——盘面全等直接恢复；差 1 格（停机期间走了一步）用
                    // movedSide 推断（持久化局面是上一手之前，移动方已走 → 轮对方，
                    // 与确认链 222 行差分推断同构）；差 ≥2（停机期间走了多步/换局）
                    // 或素材异常维持现状，bad_position 回环兜底。一次性消费——
                    // 12 帧自愈走回首次确认时素材已清，redGo 保留（自愈不清轮次）
                    restoreBoard?.let { hintBoard ->
                        restoreBoard = null
                        val moved = AssistBoard.movedSide(hintBoard, first.canonical)
                        when {
                            AssistBoard.equal(first.canonical, hintBoard) -> redGo = restoreRedGo
                            moved != null -> redGo = moved != 1
                            // else：无法推断，保持默认
                        }
                    }
                }
                return if (isNewGame) Event.NEW_GAME else Event.NEW_BOARD
            }
            unstableStreak++
            return Event.UNSTABLE
        }

        // 重开检测：与开局完全一致（32 子）视为新对局（换局可能伴随大幅子数变化，
        // 因此放在子数跳变门限之前）
        // 重开检测：开局相似（(9,7) 错位容错，依据同上）且非当前确认局面
        // （换局可能伴随大幅子数变化，因此放在子数跳变门限之前）
        if (AssistBoard.isStartLike(res.canonical) && !AssistBoard.equal(res.canonical, prev.canonical)) {
            confirmed = res
            redGo = true
            candidate = null
            candidateHits = 0
            unstableStreak = 0
            return Event.NEW_GAME
        }

        // 子数跳变门限：真实走子只 ±1（吃子 -1），检测噪声通常 ±2~3；与已确认局面
        // 相差过大只可能是局部遮挡/动画/幻觉帧，拒绝（换局走上面的重开检测）
        val tol = maxOf(2, prev.recognizedPieces / 10)
        if (abs(res.recognizedPieces - prev.recognizedPieces) > tol) {
            candidate = null
            candidateHits = 0
            unstableStreak++
            return Event.UNSTABLE
        }

        val same = AssistBoard.equal(res.canonical, prev.canonical)
        if (same) {
            candidate = null
            candidateHits = 0
            unstableStreak = 0
            return Event.SAME_BOARD
        }

        // 与候选一致则累计
        if (candidate != null && AssistBoard.equal(res.canonical, candidate!!.canonical)) {
            candidateHits++
        } else {
            candidate = res
            candidateHits = 1
        }
        if (candidateHits >= confirmCount) {
            val newBoard = candidate!!
            var toConfirm = newBoard
            // v39 走子连贯性：确认时刻做单步走子匹配。d==0（识别与唯一单步结果完全
            // 一致）锁定走子方；恰好差 1 格且唯一走子可解释 → 修正后确认（漏检帧
            // "马跳走原地残留"等被规则直接修复）。d≥2/歧义/开关关 = 行为与旧版一致。
            if (singleMoveRepair) {
                val rep = AssistBoard.repairBySingleMove(prev.canonical, newBoard.canonical)
                if (rep != null) {
                    val (fixed, side, d) = rep
                    if (d == 1) {
                        toConfirm = RecognitionResult(
                            fixed, newBoard.orientation, newBoard.issues,
                            AssistBoard.countPieces(fixed))
                        repairCount++
                    } else if (side != null) {
                        repairCount++
                    }
                    if (side != null) {
                        redGo = side != 1
                    }
                }
            }
            val moved = AssistBoard.movedSide(prev.canonical, toConfirm.canonical)
            if (moved != null) {
                redGo = moved != 1 // 移动方是红 => 下一手是黑；移动方是黑 => 下一手是红
            }
            confirmed = toConfirm
            candidate = null
            candidateHits = 0
            unstableStreak = 0
            return Event.NEW_BOARD
        }
        unstableStreak++
        return Event.UNSTABLE
    }
}
