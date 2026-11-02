package io.github.matejpopovski.stridesense.core

/** Declared in the model's output order: index `i` of the output tensor is `entries[i]`. */
enum class HumanActivity(val displayName: String) {
    WALKING("Walking"),
    WALKING_UPSTAIRS("Upstairs"),
    WALKING_DOWNSTAIRS("Downstairs"),
    SITTING("Sitting"),
    STANDING("Standing"),
    LAYING("Laying"),
}

/** One model invocation. Lives here so the pipeline can be tested without a LiteRT runtime. */
fun interface ActivityClassifier {
    /** Takes a standardized window, flat in row-major order, and returns class probabilities. */
    fun classify(window: FloatArray): FloatArray
}
