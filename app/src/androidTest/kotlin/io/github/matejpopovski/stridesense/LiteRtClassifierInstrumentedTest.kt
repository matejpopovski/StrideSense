package io.github.matejpopovski.stridesense

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.matejpopovski.stridesense.sensing.LiteRtClassifier
import kotlin.math.abs
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the packaged model through the real interpreter against outputs recorded from the Python
 * LiteRT runtime. Needs a device, because LiteRT ships only Android native libraries, so CI compiles
 * this but does not run it. Named without spaces: minSdk 26 rejects them in dex method names.
 */
@RunWith(AndroidJUnit4::class)
class LiteRtClassifierInstrumentedTest {
    @Test
    fun matchesPythonRuntimeOnGoldenWindows() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext.applicationContext as StrideApplication
        val text = instrumentation.context.assets.open("golden_windows.json")
            .bufferedReader().use { it.readText() }
        val golden = Json.parseToJsonElement(text).jsonObject

        LiteRtClassifier.fromAssets(target, target.spec).use { classifier ->
            val windows = golden.getValue("windows").jsonArray
            val expected = golden.getValue("probabilities").jsonArray
            for (index in windows.indices) {
                val window = windows[index].jsonArray.map { it.jsonPrimitive.float }.toFloatArray()
                val probabilities = classifier.classify(window)
                val truth = expected[index].jsonArray.map { it.jsonPrimitive.float }
                for (label in truth.indices) {
                    assertTrue(abs(truth[label] - probabilities[label]) < 1e-3f)
                }
            }
        }
    }
}
