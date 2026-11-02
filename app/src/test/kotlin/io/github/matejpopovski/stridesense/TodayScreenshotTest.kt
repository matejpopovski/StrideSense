package io.github.matejpopovski.stridesense

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.github.takahirom.roborazzi.captureRoboImage
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.ui.ActivityTotal
import io.github.matejpopovski.stridesense.ui.TimelineSegment
import io.github.matejpopovski.stridesense.ui.TodayScreen
import io.github.matejpopovski.stridesense.ui.TodayUiState
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

private const val MINUTE = 60_000L

/**
 * Renders the real composables through Robolectric's native graphics. The state below is written
 * by hand for the picture: it is not a recorded day, and no sensor or model is involved.
 */
@RunWith(AndroidJUnit4::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w412dp-h915dp-xxhdpi")
class TodayScreenshotTest {
    private val sampleTotals = listOf(
        ActivityTotal(HumanActivity.WALKING, 74 * MINUTE),
        ActivityTotal(HumanActivity.WALKING_UPSTAIRS, 6 * MINUTE),
        ActivityTotal(HumanActivity.WALKING_DOWNSTAIRS, 5 * MINUTE),
        ActivityTotal(HumanActivity.SITTING, 311 * MINUTE),
        ActivityTotal(HumanActivity.STANDING, 96 * MINUTE),
        ActivityTotal(HumanActivity.LAYING, 412 * MINUTE),
    )
    private val sampleTimeline = listOf(
        TimelineSegment(HumanActivity.LAYING, 0.00f, 0.29f),
        TimelineSegment(HumanActivity.STANDING, 0.29f, 0.33f),
        TimelineSegment(HumanActivity.WALKING, 0.33f, 0.36f),
        TimelineSegment(HumanActivity.SITTING, 0.36f, 0.50f),
        TimelineSegment(HumanActivity.WALKING_UPSTAIRS, 0.50f, 0.51f),
        TimelineSegment(HumanActivity.WALKING, 0.51f, 0.54f),
        TimelineSegment(HumanActivity.SITTING, 0.54f, 0.63f),
    )

    private fun capture(name: String, state: TodayUiState) {
        captureRoboImage("build/outputs/roborazzi/$name.png") {
            TodayScreen(state, onToggle = {})
        }
    }

    @Test
    fun logging() = capture(
        "today_logging",
        TodayUiState(
            running = true,
            activity = HumanActivity.WALKING,
            confidence = 0.94f,
            timeline = sampleTimeline,
            totals = sampleTotals,
        ),
    )

    @Test
    fun idle() = capture("today_idle", TodayUiState())

    @Test
    fun listening() = capture("today_listening", TodayUiState(running = true))
}
