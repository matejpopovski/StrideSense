package io.github.matejpopovski.stridesense.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Every constant the phone must share with training, read from the `preprocessing.json` that the
 * export step writes beside the model. Keeping them in data means a retrained model cannot ship
 * with the normalization of the previous one.
 */
@Serializable
data class PipelineSpec(
    @SerialName("sample_period_ns") val samplePeriodNs: Long,
    @SerialName("standard_gravity") val standardGravity: Double,
    @SerialName("window_length") val windowLength: Int,
    @SerialName("window_stride") val windowStride: Int,
    @SerialName("max_gap_ns") val maxGapNs: Long,
    @SerialName("filter_b") val filterB: List<Double>,
    @SerialName("filter_a") val filterA: List<Double>,
    @SerialName("filter_zi") val filterZi: List<Double>,
    val channels: List<String>,
    val mean: List<Double>,
    val std: List<Double>,
    val labels: List<String>,
    @SerialName("model_file") val modelFile: String,
    @SerialName("model_sha256") val modelSha256: String,
) {
    init {
        require(channels.size == CHANNELS) { "expected $CHANNELS channels, got ${channels.size}" }
        require(mean.size == CHANNELS && std.size == CHANNELS) { "normalization needs $CHANNELS values" }
        require(std.all { it > 0.0 }) { "std must be positive" }
        require(filterA.size == filterB.size && filterZi.size == filterB.size - 1) {
            "filter coefficients do not describe one filter"
        }
        require(windowStride in 1..windowLength) { "stride must fall inside the window" }
        require(labels == HumanActivity.entries.map { it.name }) {
            "model labels do not match HumanActivity: $labels"
        }
    }

    companion object {
        const val CHANNELS = 6

        private val json = Json { ignoreUnknownKeys = true }

        fun parse(text: String): PipelineSpec = json.decodeFromString(serializer(), text)
    }
}
