package io.github.matejpopovski.stridesense.core

/** One prediction after smoothing, for whoever is showing it. */
data class LivePrediction(val activity: HumanActivity, val confidence: Float, val timeMillis: Long)

/**
 * The path from sensor events to stored bouts, free of Android so it can run on the JVM. The
 * service forwards sensor events here and supplies [onBout] and [onPrediction].
 *
 * Calls must all come from one thread: the pipeline and the classifier are both single-threaded.
 */
class SensingSession(
    spec: PipelineSpec,
    private val classifier: ActivityClassifier,
    private val clockMillis: () -> Long,
    private val onBout: (Bout) -> Unit,
    private val onPrediction: (LivePrediction) -> Unit,
) {
    private val pipeline = StreamingPipeline(spec)
    private val smoother = ProbabilitySmoother()
    private val tracker = BoutTracker()

    fun onAccelerometer(timestampNs: Long, x: Float, y: Float, z: Float) =
        pipeline.onAccelerometer(timestampNs, x, y, z).forEach(::classify)

    fun onGyroscope(timestampNs: Long, x: Float, y: Float, z: Float) =
        pipeline.onGyroscope(timestampNs, x, y, z).forEach(::classify)

    private fun classify(window: DoubleArray) {
        val probabilities = classifier.classify(pipeline.standardize(window))
        val (activity, confidence) = smoother.add(probabilities).top()
        val now = clockMillis()
        onBout(tracker.update(activity, confidence, now))
        onPrediction(LivePrediction(activity, confidence, now))
    }
}
