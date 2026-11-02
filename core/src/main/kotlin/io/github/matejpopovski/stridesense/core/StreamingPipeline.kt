package io.github.matejpopovski.stridesense.core

import kotlin.math.max
import kotlin.math.min

/**
 * Turns the two unsynchronized sensor streams into model windows, one event at a time.
 *
 * This is the streaming form of `ml/preprocess.py`, which sees a whole recording at once. The
 * parity fixtures check that both produce the same windows, so a change to either has to land in
 * both. Not thread-safe: the sensor service calls it from a single handler thread.
 */
class StreamingPipeline(private val spec: PipelineSpec) {
    private val accelerometer = Resampler(spec)
    private val gyroscope = Resampler(spec)
    private val pendingAccelerometer = ArrayDeque<Event>()
    private val pendingGyroscope = ArrayDeque<Event>()
    private val alignedAccelerometer = ArrayDeque<GridSample>()
    private val alignedGyroscope = ArrayDeque<GridSample>()
    private val run = FilteredRun(spec)
    private var gridStart: Long? = null
    private var lastIndex: Long? = null

    /** Android reports m/s^2, the training data is in units of g. Returns any completed windows. */
    fun onAccelerometer(timestampNs: Long, x: Float, y: Float, z: Float): List<DoubleArray> =
        onAccelerometer(timestampNs, x.toDouble(), y.toDouble(), z.toDouble())

    fun onGyroscope(timestampNs: Long, x: Float, y: Float, z: Float): List<DoubleArray> =
        onGyroscope(timestampNs, x.toDouble(), y.toDouble(), z.toDouble())

    fun onAccelerometer(timestampNs: Long, x: Double, y: Double, z: Double): List<DoubleArray> {
        val g = spec.standardGravity
        val event = Event(timestampNs, doubleArrayOf(x / g, y / g, z / g))
        return accept(event, accelerometer, pendingAccelerometer, alignedAccelerometer)
    }

    fun onGyroscope(timestampNs: Long, x: Double, y: Double, z: Double): List<DoubleArray> {
        val event = Event(timestampNs, doubleArrayOf(x, y, z))
        return accept(event, gyroscope, pendingGyroscope, alignedGyroscope)
    }

    /** Scales a finished window into the float32 tensor the model takes, flat in row-major order. */
    fun standardize(window: DoubleArray): FloatArray {
        val channels = PipelineSpec.CHANNELS
        return FloatArray(window.size) { i ->
            val channel = i % channels
            ((window[i] - spec.mean[channel]) / spec.std[channel]).toFloat()
        }
    }

    private fun accept(
        event: Event,
        resampler: Resampler,
        pending: ArrayDeque<Event>,
        aligned: ArrayDeque<GridSample>,
    ): List<DoubleArray> {
        if (gridStart == null) {
            // The grid starts when the slower sensor wakes, so hold events until both have spoken.
            pending.addLast(event)
            if (pending.size > MAX_PENDING_EVENTS) pending.removeFirst()
            val firstAccelerometer = pendingAccelerometer.firstOrNull() ?: return emptyList()
            val firstGyroscope = pendingGyroscope.firstOrNull() ?: return emptyList()
            val start = max(firstAccelerometer.timestampNs, firstGyroscope.timestampNs)
            gridStart = start
            accelerometer.startGrid(start)
            gyroscope.startGrid(start)
            pendingAccelerometer.forEach { accelerometer.push(it, alignedAccelerometer) }
            pendingGyroscope.forEach { gyroscope.push(it, alignedGyroscope) }
            pendingAccelerometer.clear()
            pendingGyroscope.clear()
        } else {
            resampler.push(event, aligned)
        }
        return drainAligned()
    }

    private fun drainAligned(): List<DoubleArray> {
        var windows: MutableList<DoubleArray>? = null
        while (true) {
            val accel = alignedAccelerometer.firstOrNull() ?: break
            val gyro = alignedGyroscope.firstOrNull() ?: break
            when {
                accel.index < gyro.index -> alignedAccelerometer.removeFirst()

                gyro.index < accel.index -> alignedGyroscope.removeFirst()

                else -> {
                    alignedAccelerometer.removeFirst()
                    alignedGyroscope.removeFirst()
                    // A jump in the grid means a sensor dropped out. Filter state and the partial
                    // window describe the signal before the gap, so they cannot carry across it.
                    if (lastIndex?.let { it + 1 != accel.index } == true) run.reset()
                    lastIndex = accel.index
                    run.push(accel.values, gyro.values)?.let { window ->
                        val list = windows ?: mutableListOf<DoubleArray>().also { windows = it }
                        list.add(window)
                    }
                }
            }
        }
        return windows ?: emptyList()
    }

    private class Event(val timestampNs: Long, val values: DoubleArray)

    private class GridSample(val index: Long, val values: DoubleArray)

    private class Resampler(private val spec: PipelineSpec) {
        private var gridStart = 0L
        private var previous: Event? = null
        private var nextIndex = 0L

        fun startGrid(start: Long) {
            gridStart = start
        }

        fun push(event: Event, out: ArrayDeque<GridSample>) {
            val before = previous
            if (before != null && event.timestampNs <= before.timestampNs) return
            if (before == null || event.timestampNs - before.timestampNs > spec.maxGapNs) {
                nextIndex = max(0L, ceilIndex(event.timestampNs))
                if (gridTime(nextIndex) == event.timestampNs) {
                    out.addLast(GridSample(nextIndex, event.values))
                    nextIndex++
                }
            } else {
                while (gridTime(nextIndex) <= event.timestampNs) {
                    out.addLast(GridSample(nextIndex, interpolate(before, event, gridTime(nextIndex))))
                    nextIndex++
                }
            }
            previous = event
        }

        private fun gridTime(index: Long): Long = gridStart + index * spec.samplePeriodNs

        private fun ceilIndex(timestampNs: Long): Long =
            -Math.floorDiv(gridStart - timestampNs, spec.samplePeriodNs)

        private fun interpolate(from: Event, to: Event, time: Long): DoubleArray {
            val span = (to.timestampNs - from.timestampNs).toDouble()
            val offset = (time - from.timestampNs).toDouble()
            return DoubleArray(from.values.size) { axis ->
                (to.values[axis] - from.values[axis]) / span * offset + from.values[axis]
            }
        }
    }

    /** Median, then low-pass, then the sliding window, all over one contiguous run of samples. */
    private class FilteredRun(private val spec: PipelineSpec) {
        private val channels = PipelineSpec.CHANNELS
        private val filterOrder = spec.filterB.size - 1
        private val older = DoubleArray(channels)
        private val old = DoubleArray(channels)
        private val filterState = Array(channels) { DoubleArray(filterOrder) }
        private val window = DoubleArray(spec.windowLength * channels)
        private var samplesInRun = 0L
        private var samplesInWindow = 0

        fun reset() {
            samplesInRun = 0
            samplesInWindow = 0
        }

        fun push(accelerometer: DoubleArray, gyroscope: DoubleArray): DoubleArray? {
            val offset = samplesInWindow * channels
            for (channel in 0 until channels) {
                val raw = if (channel < 3) accelerometer[channel] else gyroscope[channel - 3]
                val median = if (samplesInRun >= 2) median(older[channel], old[channel], raw) else raw
                older[channel] = old[channel]
                old[channel] = raw
                window[offset + channel] = lowpass(channel, median)
            }
            samplesInRun++
            samplesInWindow++
            if (samplesInWindow < spec.windowLength) return null
            val finished = window.copyOf()
            val kept = spec.windowLength - spec.windowStride
            window.copyInto(window, 0, spec.windowStride * channels, window.size)
            samplesInWindow = kept
            return finished
        }

        /** Direct form II transposed, the structure of scipy.signal.lfilter. */
        private fun lowpass(channel: Int, input: Double): Double {
            val state = filterState[channel]
            if (samplesInRun == 0L) {
                // Starting from zero would ring for several samples on the 1 g of gravity.
                for (i in state.indices) state[i] = spec.filterZi[i] * input
            }
            val output = spec.filterB[0] * input + state[0]
            for (i in 0 until filterOrder) {
                val next = if (i + 1 < filterOrder) state[i + 1] else 0.0
                state[i] = spec.filterB[i + 1] * input - spec.filterA[i + 1] * output + next
            }
            return output
        }

        private fun median(a: Double, b: Double, c: Double): Double = max(min(a, b), min(max(a, b), c))
    }

    private companion object {
        // Roughly 20 seconds at the sensor rate: a sensor silent for longer than that is not going
        // to start, and this bounds what a dead gyroscope can cost in memory.
        const val MAX_PENDING_EVENTS = 1024
    }
}
