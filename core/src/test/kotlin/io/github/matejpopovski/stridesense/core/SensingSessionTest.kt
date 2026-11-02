package io.github.matejpopovski.stridesense.core

import io.github.matejpopovski.stridesense.core.HumanActivity.LAYING
import io.github.matejpopovski.stridesense.core.HumanActivity.WALKING
import kotlin.test.Test
import kotlin.test.assertEquals

class SensingSessionTest {
    private val spec = identitySpec(windowLength = 4, windowStride = 2)
    private val classes = HumanActivity.entries.size

    private fun probabilities(activity: HumanActivity) =
        FloatArray(classes).also { it[activity.ordinal] = 1f }

    private inner class Harness(answers: List<HumanActivity>) {
        var now = 100_000L
        val bouts = mutableListOf<Bout>()
        val predictions = mutableListOf<LivePrediction>()
        val windows = mutableListOf<FloatArray>()
        private val queue = ArrayDeque(answers)
        val session = SensingSession(
            spec = spec,
            classifier = { window ->
                windows += window
                now += BoutTracker.STRIDE_MILLIS
                probabilities(queue.removeFirst())
            },
            clockMillis = { now },
            onBout = { bouts += it },
            onPrediction = { predictions += it },
        )

        fun feed(samples: Int) {
            for (i in 0 until samples) {
                val time = i * PERIOD_NS
                session.onAccelerometer(time, 0f, 0f, 9.80665f)
                session.onGyroscope(time, 0f, 0f, 0f)
            }
        }
    }

    @Test
    fun `no prediction is made before the first window fills`() {
        val harness = Harness(emptyList())
        harness.feed(3)
        assertEquals(0, harness.predictions.size)
    }

    @Test
    fun `the classifier receives the standardized flat window`() {
        val harness = Harness(listOf(WALKING))
        harness.feed(4)
        val window = harness.windows.single()
        assertEquals(4 * PipelineSpec.CHANNELS, window.size)
        assertEquals(1f, window[2], 1e-6f)
    }

    @Test
    fun `each window produces one prediction and one bout update`() {
        val harness = Harness(List(3) { WALKING })
        harness.feed(8)
        assertEquals(3, harness.predictions.size)
        assertEquals(3, harness.bouts.size)
    }

    @Test
    fun `repeated windows of one activity keep extending one bout`() {
        val harness = Harness(List(3) { WALKING })
        harness.feed(8)
        assertEquals(1, harness.bouts.map { it.startMillis }.distinct().size)
        assertEquals(3, harness.bouts.last().windows)
    }

    @Test
    fun `a brief misclassification does not end the bout`() {
        val harness = Harness(listOf(WALKING, WALKING, LAYING, WALKING))
        harness.feed(10)
        assertEquals(setOf(WALKING), harness.bouts.map { it.activity }.toSet())
    }

    @Test
    fun `a sustained change opens a new bout`() {
        val harness = Harness(listOf(WALKING, WALKING, LAYING, LAYING, LAYING))
        harness.feed(12)
        assertEquals(WALKING, harness.bouts.first().activity)
        assertEquals(LAYING, harness.bouts.last().activity)
    }
}
