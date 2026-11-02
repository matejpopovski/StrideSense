package io.github.matejpopovski.stridesense

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.ui.TodayScreen
import io.github.matejpopovski.stridesense.ui.TodayUiState
import io.github.matejpopovski.stridesense.ui.formatDuration
import kotlin.test.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TodayScreenTest {
    @get:Rule
    val compose = createComposeRule()

    private fun show(state: TodayUiState, onToggle: () -> Unit = {}) {
        compose.setContent { TodayScreen(state, onToggle) }
    }

    @Test
    fun `an idle screen says it is not logging`() {
        show(TodayUiState())
        compose.onNodeWithText("NOT LOGGING").assertIsDisplayed()
        compose.onNodeWithText("START LOGGING").performScrollTo().assertIsDisplayed()
    }

    @Test
    fun `a running screen shows the activity and its confidence`() {
        show(TodayUiState(running = true, activity = HumanActivity.WALKING_UPSTAIRS, confidence = 0.87f))
        compose.onNodeWithText("UPSTAIRS").assertIsDisplayed()
        compose.onNodeWithText("87% confidence").assertIsDisplayed()
    }

    @Test
    fun `running before the first window says it is listening`() {
        show(TodayUiState(running = true))
        compose.onNodeWithText("LISTENING").assertIsDisplayed()
        compose.onNodeWithText("STOP LOGGING").performScrollTo().assertIsDisplayed()
    }

    @Test
    fun `the button reports a toggle`() {
        var toggles = 0
        show(TodayUiState()) { toggles++ }
        compose.onNodeWithText("START LOGGING").performScrollTo().performClick()
        assertEquals(1, toggles)
    }

    @Test
    fun `a sensing problem is shown`() {
        show(TodayUiState(problem = SensingProblem.NO_SENSORS))
        compose.onNodeWithText("This device lacks an accelerometer or gyroscope.").assertIsDisplayed()
    }

    @Test
    fun `the waist placement caveat is always on screen`() {
        show(TodayUiState())
        compose
            .onNodeWithText(
                "Trained on phones worn sideways at the waist. Expect poor results in a pocket or hand. Real-world accuracy is unmeasured.",
            )
            .performScrollTo()
            .assertIsDisplayed()
    }

    @Test
    fun `durations read in minutes, then hours and minutes`() {
        assertEquals("0m", formatDuration(59_000))
        assertEquals("59m", formatDuration(59 * 60_000L))
        assertEquals("1h 0m", formatDuration(60 * 60_000L))
        assertEquals("2h 35m", formatDuration(155 * 60_000L))
    }
}
