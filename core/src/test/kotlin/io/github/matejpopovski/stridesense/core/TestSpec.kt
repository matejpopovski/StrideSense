package io.github.matejpopovski.stridesense.core

/** A spec whose filter is the identity, so tests can reason about samples directly. */
fun identitySpec(windowLength: Int = 8, windowStride: Int = 4, maxGapNs: Long = 200_000_000) = PipelineSpec(
    samplePeriodNs = 20_000_000,
    standardGravity = 9.80665,
    windowLength = windowLength,
    windowStride = windowStride,
    maxGapNs = maxGapNs,
    filterB = listOf(1.0, 0.0),
    filterA = listOf(1.0, 0.0),
    filterZi = listOf(0.0),
    channels = listOf("ax", "ay", "az", "gx", "gy", "gz"),
    mean = listOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    std = listOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    labels = HumanActivity.entries.map { it.name },
    modelFile = "model.tflite",
    modelSha256 = "",
)

const val PERIOD_NS = 20_000_000L
