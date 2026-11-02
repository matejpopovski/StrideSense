package io.github.matejpopovski.stridesense

import android.app.Application
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.core.PipelineSpec
import io.github.matejpopovski.stridesense.data.BoutRepository
import io.github.matejpopovski.stridesense.data.StrideDatabase
import io.github.matejpopovski.stridesense.sensing.ClosableClassifier
import io.github.matejpopovski.stridesense.sensing.LiteRtClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow

enum class SensingProblem { NO_SENSORS, SENSOR_REFUSED }

data class LiveStatus(
    val running: Boolean = false,
    val activity: HumanActivity? = null,
    val confidence: Float = 0f,
    val problem: SensingProblem? = null,
)

class StrideApplication : Application() {
    val spec: PipelineSpec by lazy {
        val text = assets.open(SPEC_ASSET).bufferedReader().use { it.readText() }
        PipelineSpec.parse(text)
    }

    private val database by lazy { StrideDatabase.open(this) }

    val bouts: BoutRepository by lazy { BoutRepository(database.bouts()) }

    val liveStatus = MutableStateFlow(LiveStatus())

    /** Outlives the service, so a bout written as logging stops still reaches the database. */
    val writeScope = CoroutineScope(SupervisorJob() + Dispatchers.IO.limitedParallelism(1))

    /** Replaced in Robolectric tests, whose JVM cannot load LiteRT's Android native library. */
    var classifierFactory: () -> ClosableClassifier = { LiteRtClassifier.fromAssets(this, spec) }

    companion object {
        const val SPEC_ASSET = "preprocessing.json"
    }
}
