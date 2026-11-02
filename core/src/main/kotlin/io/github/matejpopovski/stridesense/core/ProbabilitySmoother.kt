package io.github.matejpopovski.stridesense.core

/**
 * Averages class probabilities over the latest windows, so one odd window does not split a bout.
 * Averaging keeps more information than a majority vote over labels: a confident window can
 * outweigh two unsure ones.
 */
class ProbabilitySmoother(private val windows: Int = DEFAULT_WINDOWS) {
    private val recent = ArrayDeque<FloatArray>()

    init {
        require(windows >= 1) { "need at least one window" }
    }

    fun add(probabilities: FloatArray): FloatArray {
        recent.addLast(probabilities)
        if (recent.size > windows) recent.removeFirst()
        return FloatArray(probabilities.size) { i -> recent.map { it[i] }.average().toFloat() }
    }

    fun reset() = recent.clear()

    companion object {
        const val DEFAULT_WINDOWS = 3
    }
}

/** The most probable class and its probability. */
fun FloatArray.top(): Pair<HumanActivity, Float> {
    val best = indices.maxBy { this[it] }
    return HumanActivity.entries[best] to this[best]
}
