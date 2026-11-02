package io.github.matejpopovski.stridesense.sensing

import android.content.Context
import io.github.matejpopovski.stridesense.core.ActivityClassifier
import io.github.matejpopovski.stridesense.core.PipelineSpec
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter

/** An [ActivityClassifier] that must be released, because it owns a native interpreter. */
interface ClosableClassifier :
    ActivityClassifier,
    AutoCloseable

/**
 * Runs the exported model. Not thread-safe, and it does not need to be: the sensor service calls
 * it from its one handler thread. One thread for the interpreter too, because a window arrives
 * every 1.28 s and waking more cores to save milliseconds would only cost battery.
 */
class LiteRtClassifier(model: ByteBuffer, spec: PipelineSpec) : ClosableClassifier {
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(1))
    private val input = ByteBuffer
        .allocateDirect(spec.windowLength * PipelineSpec.CHANNELS * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val output = arrayOf(FloatArray(spec.labels.size))

    override fun classify(window: FloatArray): FloatArray {
        input.rewind()
        input.asFloatBuffer().put(window)
        interpreter.run(input, output)
        return output[0].copyOf()
    }

    override fun close() = interpreter.close()

    companion object {
        fun fromAssets(context: Context, spec: PipelineSpec): LiteRtClassifier {
            val bytes = context.assets.open(spec.modelFile).use { it.readBytes() }
            // Reading into memory rather than mapping the asset also works if the build compresses
            // it, and the model is small enough for that not to matter.
            val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            model.put(bytes).rewind()
            return LiteRtClassifier(model, spec)
        }
    }
}
