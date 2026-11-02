package io.github.matejpopovski.stridesense.core

import io.github.matejpopovski.stridesense.core.HumanActivity.SITTING
import io.github.matejpopovski.stridesense.core.HumanActivity.WALKING
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class BoutsTest {
    private val stride = BoutTracker.STRIDE_MILLIS

    @Test
    fun `the first prediction opens a bout covering one stride`() {
        val bout = BoutTracker().update(WALKING, 0.9f, 10_000)
        assertEquals(10_000 - stride, bout.startMillis)
        assertEquals(10_000, bout.endMillis)
    }

    @Test
    fun `the same activity extends the open bout and keeps its key`() {
        val tracker = BoutTracker()
        val first = tracker.update(WALKING, 0.8f, 10_000)
        val second = tracker.update(WALKING, 0.6f, 10_000 + stride)
        assertEquals(first.startMillis, second.startMillis)
        assertEquals(2, second.windows)
        assertEquals(0.7, second.meanConfidence, 1e-6)
    }

    @Test
    fun `a change of activity starts where the previous bout ended`() {
        val tracker = BoutTracker()
        val walking = tracker.update(WALKING, 0.9f, 10_000)
        val sitting = tracker.update(SITTING, 0.9f, 10_000 + stride)
        assertEquals(SITTING, sitting.activity)
        assertEquals(walking.endMillis, sitting.startMillis)
    }

    @Test
    fun `silence past the limit opens a new bout rather than bridging it`() {
        val tracker = BoutTracker()
        val before = tracker.update(WALKING, 0.9f, 10_000)
        val after = tracker.update(WALKING, 0.9f, 10_000 + BoutTracker.MAX_SILENCE_MILLIS + 1)
        assertEquals(1, after.windows)
        assert(after.startMillis > before.endMillis)
    }

    @Test
    fun `silence up to the limit is bridged`() {
        val tracker = BoutTracker()
        tracker.update(WALKING, 0.9f, 10_000)
        val after = tracker.update(WALKING, 0.9f, 10_000 + BoutTracker.MAX_SILENCE_MILLIS)
        assertEquals(2, after.windows)
    }

    @Test
    fun `reset forgets the open bout`() {
        val tracker = BoutTracker()
        tracker.update(WALKING, 0.9f, 10_000)
        tracker.reset()
        assertEquals(1, tracker.update(WALKING, 0.9f, 11_000).windows)
    }

    @Test
    fun `daily totals add up per activity`() {
        val bouts = listOf(
            Bout(WALKING, 100, 300, 1, 1.0),
            Bout(SITTING, 300, 400, 1, 1.0),
            Bout(WALKING, 400, 450, 1, 1.0),
        )
        val totals = dailyTotals(bouts, 0, 1_000)
        assertEquals(250, totals[WALKING])
        assertEquals(100, totals[SITTING])
        assertEquals(0, totals[HumanActivity.LAYING])
    }

    @Test
    fun `a bout crossing midnight counts only its part in the day`() {
        val bouts = listOf(Bout(WALKING, 900, 1_100, 1, 1.0))
        assertEquals(100, dailyTotals(bouts, 0, 1_000)[WALKING])
        assertEquals(100, dailyTotals(bouts, 1_000, 2_000)[WALKING])
    }

    @Test
    fun `a bout outside the day clips to nothing`() {
        assertNull(Bout(WALKING, 2_000, 3_000, 1, 1.0).clippedTo(0, 1_000))
    }

    @Test
    fun `smoothing averages over the latest windows only`() {
        val smoother = ProbabilitySmoother(windows = 2)
        smoother.add(floatArrayOf(1f, 0f))
        smoother.add(floatArrayOf(0f, 1f))
        val latest = smoother.add(floatArrayOf(0f, 1f))
        assertEquals(0f, latest[0], 1e-6f)
        assertEquals(1f, latest[1], 1e-6f)
    }

    @Test
    fun `one odd window does not outvote two confident ones`() {
        val smoother = ProbabilitySmoother()
        smoother.add(floatArrayOf(0.9f, 0.1f, 0f, 0f, 0f, 0f))
        smoother.add(floatArrayOf(0.9f, 0.1f, 0f, 0f, 0f, 0f))
        val (activity, confidence) = smoother.add(floatArrayOf(0.4f, 0.6f, 0f, 0f, 0f, 0f)).top()
        assertEquals(WALKING, activity)
        assertEquals(0.733f, confidence, 1e-3f)
    }

    @Test
    fun `a spec with reordered labels is rejected`() {
        val labels = identitySpec().labels.reversed()
        assertFailsWith<IllegalArgumentException> { identitySpec().copy(labels = labels) }
    }

    @Test
    fun `a spec with a mismatched filter is rejected`() {
        assertFailsWith<IllegalArgumentException> { identitySpec().copy(filterZi = emptyList()) }
    }

    @Test
    fun `a spec with zero std is rejected`() {
        val std = listOf(1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
        assertFailsWith<IllegalArgumentException> { identitySpec().copy(std = std) }
    }
}
