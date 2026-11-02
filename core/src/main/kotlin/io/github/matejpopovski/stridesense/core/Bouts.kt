package io.github.matejpopovski.stridesense.core

/** A stretch of one activity. [startMillis] is unique and doubles as its key in storage. */
data class Bout(
    val activity: HumanActivity,
    val startMillis: Long,
    val endMillis: Long,
    val windows: Int,
    val confidenceSum: Double,
) {
    val durationMillis: Long get() = endMillis - startMillis
    val meanConfidence: Double get() = if (windows == 0) 0.0 else confidenceSum / windows
}

/**
 * Folds predictions into bouts. Returns the bout each prediction extended or opened, which is the
 * row to persist, so a crash loses at most one window.
 */
class BoutTracker(
    private val strideMillis: Long = STRIDE_MILLIS,
    private val maxSilenceMillis: Long = MAX_SILENCE_MILLIS,
) {
    private var current: Bout? = null

    fun update(activity: HumanActivity, confidence: Float, timeMillis: Long): Bout {
        val open = current
        val updated = when {
            open == null || timeMillis - open.endMillis > maxSilenceMillis ->
                Bout(activity, timeMillis - strideMillis, timeMillis, 1, confidence.toDouble())

            open.activity != activity ->
                Bout(activity, open.endMillis, timeMillis, 1, confidence.toDouble())

            else -> open.copy(
                endMillis = timeMillis,
                windows = open.windows + 1,
                confidenceSum = open.confidenceSum + confidence,
            )
        }
        current = updated
        return updated
    }

    fun reset() {
        current = null
    }

    companion object {
        /** 64 samples at 50 Hz: one prediction accounts for this much of the day. */
        const val STRIDE_MILLIS = 1_280L

        /** Past this the service was stopped or the sensors stalled, so the bout does not bridge it. */
        const val MAX_SILENCE_MILLIS = 10_000L
    }
}

/** What a bout contributes to one calendar day. */
fun Bout.clippedTo(dayStartMillis: Long, dayEndMillis: Long): Bout? {
    val start = maxOf(startMillis, dayStartMillis)
    val end = minOf(endMillis, dayEndMillis)
    return if (end > start) copy(startMillis = start, endMillis = end) else null
}

/** Milliseconds per activity for one day. Bouts crossing midnight count only their part in it. */
fun dailyTotals(bouts: List<Bout>, dayStartMillis: Long, dayEndMillis: Long): Map<HumanActivity, Long> {
    val totals = HumanActivity.entries.associateWith { 0L }.toMutableMap()
    for (bout in bouts) {
        val clipped = bout.clippedTo(dayStartMillis, dayEndMillis) ?: continue
        totals[bout.activity] = totals.getValue(bout.activity) + clipped.durationMillis
    }
    return totals
}
