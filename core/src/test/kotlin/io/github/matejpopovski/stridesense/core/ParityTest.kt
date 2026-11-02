package io.github.matejpopovski.stridesense.core

import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

private val repoRoot = File(System.getProperty("stridesense.root"))
private val shippedSpec =
    PipelineSpec.parse(repoRoot.resolve("app/src/main/assets/preprocessing.json").readText())

private fun doubles(array: JsonArray) = DoubleArray(array.size) { array[it].jsonPrimitive.double }

private class SensorEvents(json: JsonObject) {
    val times = json.getValue("t").jsonArray.map { it.jsonPrimitive.long }
    val values = json.getValue("v").jsonArray.map { doubles(it.jsonArray) }
}

private class Fixture(file: File) {
    val root = Json.parseToJsonElement(file.readText()).jsonObject
    val name = root.getValue("name").jsonPrimitive.content
    val accelerometer = SensorEvents(root.getValue("accelerometer").jsonObject)
    val gyroscope = SensorEvents(root.getValue("gyroscope").jsonObject)
    val windows = root.getValue("windows").jsonArray.map { doubles(it.jsonArray) }
    val modelInput = root.getValue("model_input").jsonArray.map { doubles(it.jsonArray) }

    /**
     * Replays the events in time order across both sensors, as a phone would deliver them. Python
     * saw each stream whole, so the streaming port has to reach the same answer with less to go on.
     */
    fun run(spec: PipelineSpec): List<DoubleArray> {
        val pipeline = StreamingPipeline(spec)
        val windows = mutableListOf<DoubleArray>()
        var a = 0
        var g = 0
        while (a < accelerometer.times.size || g < gyroscope.times.size) {
            val takeAccelerometer = g >= gyroscope.times.size ||
                (a < accelerometer.times.size && accelerometer.times[a] <= gyroscope.times[g])
            if (takeAccelerometer) {
                val (x, y, z) = accelerometer.values[a]
                windows += pipeline.onAccelerometer(accelerometer.times[a++], x, y, z)
            } else {
                val (x, y, z) = gyroscope.values[g]
                windows += pipeline.onGyroscope(gyroscope.times[g++], x, y, z)
            }
        }
        return windows
    }
}

private val fixtures = repoRoot.resolve("fixtures/parity").listFiles { file -> file.extension == "json" }!!
    .sortedBy { it.name }
    .map(::Fixture)

private fun worstDifference(expected: List<DoubleArray>, actual: List<DoubleArray>): Double {
    if (expected.size != actual.size) return Double.POSITIVE_INFINITY
    var worst = 0.0
    for ((want, got) in expected.zip(actual)) {
        if (want.size != got.size) return Double.POSITIVE_INFINITY
        for (i in want.indices) worst = max(worst, abs(want[i] - got[i]))
    }
    return worst
}

class ParityTest {
    @Test
    fun `the fixtures cover the failure modes they are named for`() {
        assertEquals(5, fixtures.size)
        assertTrue(fixtures.all { it.windows.size >= 2 }, "every fixture must yield several windows")
    }

    @Test
    fun `kotlin windows match the python reference`() {
        val differences = fixtures.associate { it.name to worstDifference(it.windows, it.run(shippedSpec)) }
        differences.forEach { (name, worst) ->
            assertTrue(worst < WINDOW_TOLERANCE, "$name differs from Python by $worst")
        }
        System.getenv("STRIDESENSE_PARITY_OUT")?.let { path ->
            val entries = differences.entries.joinToString(",\n") { (name, worst) ->
                val windows = fixtures.single { it.name == name }.windows.size
                """    "$name": {"windows": $windows, "max_abs_difference": $worst}"""
            }
            File(
                path,
            ).writeText("{\n  \"tolerance\": $WINDOW_TOLERANCE,\n  \"fixtures\": {\n$entries\n  }\n}\n")
        }
    }

    @Test
    fun `the model input matches python to float precision`() {
        for (fixture in fixtures) {
            val pipeline = StreamingPipeline(shippedSpec)
            val actual = fixture.run(shippedSpec).map { window ->
                val floats = pipeline.standardize(window)
                DoubleArray(floats.size) { floats[it].toDouble() }
            }
            assertTrue(
                worstDifference(fixture.modelInput, actual) < INPUT_TOLERANCE,
                "${fixture.name} model input differs",
            )
        }
    }

    @Test
    fun `each deliberate porting mistake is caught by the fixtures`() {
        val mutants = mapOf(
            "gravity rounded to 9.81" to shippedSpec.copy(standardGravity = 9.81),
            "stride off by one" to shippedSpec.copy(windowStride = shippedSpec.windowStride - 1),
            "window off by one" to shippedSpec.copy(windowLength = shippedSpec.windowLength - 1),
            "filter starts from zero" to
                shippedSpec.copy(filterZi = shippedSpec.filterZi.map { 0.0 }),
            "filter coefficient sign flipped" to shippedSpec.copy(
                filterA = shippedSpec.filterA.mapIndexed { i, a -> if (i == 2) -a else a },
            ),
            "sampling period 1 ms long" to shippedSpec.copy(samplePeriodNs = 21_000_000),
            "gap limit tripled" to shippedSpec.copy(maxGapNs = shippedSpec.maxGapNs * 3),
        )
        for ((name, spec) in mutants) {
            val worst = fixtures.maxOf { worstDifference(it.windows, it.run(spec)) }
            assertTrue(worst > WINDOW_TOLERANCE, "'$name' went unnoticed: worst difference $worst")
        }
    }

    @Test
    fun `swapped normalization is caught at the model input`() {
        val swapped = shippedSpec.copy(mean = shippedSpec.mean.reversed())
        val fixture = fixtures.first()
        val pipeline = StreamingPipeline(swapped)
        val actual = fixture.run(shippedSpec).map { window ->
            val floats = pipeline.standardize(window)
            DoubleArray(floats.size) { floats[it].toDouble() }
        }
        assertTrue(worstDifference(fixture.modelInput, actual) > INPUT_TOLERANCE)
    }

    @Test
    fun `the shipped spec records the file the app loads`() {
        assertTrue(repoRoot.resolve("app/src/main/assets/${shippedSpec.modelFile}").isFile)
    }

    private companion object {
        const val WINDOW_TOLERANCE = 1e-9
        const val INPUT_TOLERANCE = 1e-6
    }
}
