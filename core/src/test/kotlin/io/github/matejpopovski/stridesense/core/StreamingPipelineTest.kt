package io.github.matejpopovski.stridesense.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StreamingPipelineTest {
    private fun StreamingPipeline.feed(index: Int, gyro: Double = index.toDouble()): List<DoubleArray> {
        val time = index * PERIOD_NS
        return onAccelerometer(time, 0.0, 0.0, 0.0) + onGyroscope(time, gyro, 0.0, 0.0)
    }

    private fun gyroX(window: DoubleArray) = window.filterIndexed { i, _ -> i % 6 == 3 }

    @Test
    fun `accelerometer is converted from metres per second squared to g`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 4, windowStride = 4))
        var window: DoubleArray? = null
        for (i in 0 until 4) {
            val time = i * PERIOD_NS
            window = (
                pipeline.onAccelerometer(time, 9.80665, 0.0, 0.0) +
                    pipeline.onGyroscope(time, 0.0, 0.0, 0.0)
                ).firstOrNull() ?: window
        }
        assertEquals(1.0, window!![0], 1e-12)
    }

    @Test
    fun `no window is emitted before the window fills`() {
        val pipeline = StreamingPipeline(identitySpec())
        for (i in 0 until 7) assertTrue(pipeline.feed(i).isEmpty())
        assertEquals(1, pipeline.feed(7).size)
    }

    @Test
    fun `consecutive windows overlap by window length minus stride`() {
        val pipeline = StreamingPipeline(identitySpec())
        val windows = (0 until 12).flatMap { pipeline.feed(it) }
        assertEquals(2, windows.size)
        assertEquals(gyroX(windows[0]).drop(4), gyroX(windows[1]).take(4))
    }

    @Test
    fun `samples between events are interpolated onto the grid`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 2, windowStride = 2))
        val windows = mutableListOf<DoubleArray>()
        windows += pipeline.onGyroscope(0, 0.0, 0.0, 0.0)
        windows += pipeline.onGyroscope(40_000_000, 4.0, 0.0, 0.0)
        windows += pipeline.onAccelerometer(0, 0.0, 0.0, 0.0)
        windows += pipeline.onAccelerometer(40_000_000, 0.0, 0.0, 0.0)
        assertEquals(listOf(0.0, 2.0), gyroX(windows.single()))
    }

    @Test
    fun `the grid starts when the later sensor starts`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 2, windowStride = 2))
        val windows = mutableListOf<DoubleArray>()
        for (i in 0..5) windows += pipeline.onGyroscope(i * PERIOD_NS, i.toDouble(), 0.0, 0.0)
        for (i in 3..5) windows += pipeline.onAccelerometer(i * PERIOD_NS, 0.0, 0.0, 0.0)
        assertEquals(listOf(3.0, 4.0), gyroX(windows.first()))
    }

    @Test
    fun `sensors reporting in either order produce the same windows`() {
        val gyroFirst = StreamingPipeline(identitySpec(windowLength = 4, windowStride = 2))
        val accelFirst = StreamingPipeline(identitySpec(windowLength = 4, windowStride = 2))
        val one = (0 until 6).flatMap { i ->
            val t = i * PERIOD_NS + 3_000_000
            gyroFirst.onGyroscope(t, i.toDouble(), 0.0, 0.0) +
                gyroFirst.onAccelerometer(t, i.toDouble(), 0.0, 0.0)
        }
        val two = (0 until 6).flatMap { i ->
            val t = i * PERIOD_NS + 3_000_000
            accelFirst.onAccelerometer(t, i.toDouble(), 0.0, 0.0) +
                accelFirst.onGyroscope(t, i.toDouble(), 0.0, 0.0)
        }
        assertEquals(one.size, two.size)
        one.zip(two).forEach { (a, b) -> assertContentEquals(a, b) }
    }

    @Test
    fun `a sensor gap restarts the window instead of interpolating across it`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 4, windowStride = 4))
        val windows = mutableListOf<DoubleArray>()
        (0 until 3).forEach { windows += pipeline.feed(it, gyro = 1.0) }
        (50 until 54).forEach { windows += pipeline.feed(it, gyro = 7.0) }
        assertEquals(listOf(7.0, 7.0, 7.0, 7.0), gyroX(windows.single()))
    }

    @Test
    fun `a gap of exactly the limit is still ordinary jitter`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 4, windowStride = 4))
        val windows = mutableListOf<DoubleArray>()
        (0 until 3).forEach { windows += pipeline.feed(it, gyro = 1.0) }
        windows += pipeline.feed(12, gyro = 1.0)
        assertEquals(listOf(1.0, 1.0, 1.0, 1.0), gyroX(windows.first()))
    }

    @Test
    fun `a gap one nanosecond past the limit breaks the run`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 4, windowStride = 4))
        val windows = mutableListOf<DoubleArray>()
        (0 until 3).forEach { windows += pipeline.feed(it, gyro = 1.0) }
        val late = 2 * PERIOD_NS + 200_000_001
        windows += pipeline.onAccelerometer(late, 0.0, 0.0, 0.0)
        windows += pipeline.onGyroscope(late, 1.0, 0.0, 0.0)
        assertTrue(windows.isEmpty())
    }

    @Test
    fun `the filter restarts settled on the first sample after a gap`() {
        val alpha = 0.5
        val spec = identitySpec(windowLength = 4, windowStride = 4).copy(
            filterB = listOf(alpha, 0.0),
            filterA = listOf(1.0, -alpha),
            filterZi = listOf(1.0 - alpha),
        )
        val pipeline = StreamingPipeline(spec)
        val windows = mutableListOf<DoubleArray>()
        (0 until 3).forEach { windows += pipeline.feed(it, gyro = 1.0) }
        (50 until 54).forEach { windows += pipeline.feed(it, gyro = 7.0) }
        assertEquals(listOf(7.0, 7.0, 7.0, 7.0), gyroX(windows.single()))
    }

    @Test
    fun `an event that does not advance time is ignored`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 2, windowStride = 2))
        val windows = mutableListOf<DoubleArray>()
        windows += pipeline.feed(0)
        windows += pipeline.onGyroscope(0, 99.0, 0.0, 0.0)
        windows += pipeline.feed(1)
        assertEquals(listOf(0.0, 1.0), gyroX(windows.single()))
    }

    @Test
    fun `the median removes a single sample spike`() {
        val pipeline = StreamingPipeline(identitySpec(windowLength = 5, windowStride = 5))
        val signal = listOf(1.0, 1.0, 50.0, 1.0, 1.0)
        val windows = signal.flatMapIndexed { i, value -> pipeline.feed(i, value) }
        assertEquals(listOf(1.0, 1.0, 1.0, 1.0, 1.0), gyroX(windows.single()))
    }

    @Test
    fun `a dead gyroscope cannot grow memory without bound`() {
        val pipeline = StreamingPipeline(identitySpec())
        for (i in 0 until 100_000) {
            assertTrue(pipeline.onAccelerometer(i * PERIOD_NS, 0.0, 0.0, 0.0).isEmpty())
        }
    }

    @Test
    fun `standardize subtracts the mean and divides by the std per channel`() {
        val spec = identitySpec().copy(
            mean = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            std = listOf(2.0, 2.0, 2.0, 2.0, 2.0, 4.0),
        )
        val scaled = StreamingPipeline(spec).standardize(doubleArrayOf(3.0, 2.0, 3.0, 4.0, 5.0, 14.0))
        assertContentEquals(floatArrayOf(1f, 0f, 0f, 0f, 0f, 2f), scaled)
    }
}
