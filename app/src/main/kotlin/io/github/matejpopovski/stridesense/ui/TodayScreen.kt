package io.github.matejpopovski.stridesense.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.matejpopovski.stridesense.R
import io.github.matejpopovski.stridesense.SensingProblem
import io.github.matejpopovski.stridesense.core.HumanActivity
import kotlin.math.roundToInt

private val Ink = Color(0xFF000000)
private val Chalk = Color(0xFFFFFFFF)
private val Volt = Color(0xFFC6FF00)
private val Track = Color(0xFF2A2A2A)
private val Muted = Color(0xFFB8B8B8)

/** Six hues that stay apart on black. The list below repeats each name, so color is never alone. */
private val HumanActivity.color: Color
    get() = when (this) {
        HumanActivity.WALKING -> Volt
        HumanActivity.WALKING_UPSTAIRS -> Color(0xFFFF6D00)
        HumanActivity.WALKING_DOWNSTAIRS -> Color(0xFF00E5FF)
        HumanActivity.SITTING -> Color(0xFFFF4081)
        HumanActivity.STANDING -> Chalk
        HumanActivity.LAYING -> Color(0xFF7C8CFF)
    }

@Composable
fun TodayScreen(state: TodayUiState, onToggle: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Ink)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        Text(
            stringResource(R.string.app_name).uppercase(),
            color = Muted,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 3.sp,
        )
        Spacer(Modifier.height(20.dp))
        Hero(state)
        Spacer(Modifier.height(28.dp))
        Heading(stringResource(R.string.section_timeline))
        Timeline(state.timeline)
        Spacer(Modifier.height(28.dp))
        Heading(stringResource(R.string.section_totals))
        Totals(state.totals)
        Spacer(Modifier.height(24.dp))
        Button(
            onClick = onToggle,
            modifier = Modifier.fillMaxWidth().height(60.dp),
            shape = RectangleShape,
            colors = ButtonDefaults.buttonColors(
                containerColor = if (state.running) Chalk else Volt,
                contentColor = Ink,
            ),
        ) {
            Text(
                stringResource(if (state.running) R.string.action_stop else R.string.action_start),
                fontWeight = FontWeight.Black,
                fontSize = 17.sp,
                letterSpacing = 1.sp,
            )
        }
        Spacer(Modifier.height(16.dp))
        Text(stringResource(R.string.placement_note), color = Muted, fontSize = 13.sp)
    }
}

@Composable
private fun Hero(state: TodayUiState) {
    val activity = state.activity
    val headline = when {
        activity != null && state.running -> activity.displayName.uppercase()
        state.running -> stringResource(R.string.status_waiting)
        else -> stringResource(R.string.status_idle)
    }
    Text(
        headline,
        color = activity?.takeIf { state.running }?.color ?: Chalk,
        fontSize = 52.sp,
        lineHeight = 54.sp,
        fontWeight = FontWeight.Black,
        fontStyle = FontStyle.Italic,
    )
    if (activity != null && state.running) {
        Spacer(Modifier.height(10.dp))
        Bar(state.confidence, activity.color)
        Spacer(Modifier.height(6.dp))
        Text(
            stringResource(R.string.confidence_value, (state.confidence * 100).roundToInt()),
            color = Muted,
            fontSize = 14.sp,
        )
    }
    state.problem?.let { problem ->
        Spacer(Modifier.height(10.dp))
        Text(
            stringResource(
                when (problem) {
                    SensingProblem.NO_SENSORS -> R.string.problem_no_sensors
                    SensingProblem.SENSOR_REFUSED -> R.string.problem_sensor_refused
                },
            ),
            color = Color(0xFFFF8A80),
            fontSize = 14.sp,
        )
    }
}

@Composable
private fun Heading(text: String) {
    Text(
        text,
        color = Muted,
        fontSize = 12.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = 2.sp,
        modifier = Modifier.padding(bottom = 8.dp),
    )
}

@Composable
private fun Timeline(segments: List<TimelineSegment>) {
    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(36.dp)
            .background(Track)
            .semantics { contentDescription = "Timeline of the day from midnight to midnight" },
    ) {
        for (segment in segments) {
            val left = segment.startFraction * size.width
            // A short bout must stay visible: at one pixel per minute it would otherwise vanish.
            val width = ((segment.endFraction - segment.startFraction) * size.width)
                .coerceAtLeast(2.dp.toPx())
            drawRect(segment.activity.color, Offset(left, 0f), Size(width, size.height))
        }
    }
    Spacer(Modifier.height(4.dp))
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        listOf("00", "06", "12", "18", "24").forEach { Text(it, color = Muted, fontSize = 11.sp) }
    }
}

@Composable
private fun Totals(totals: List<ActivityTotal>) {
    val longest = totals.maxOf { it.millis }.coerceAtLeast(1L)
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        for (total in totals) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    total.activity.displayName,
                    color = Chalk,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.width(104.dp),
                )
                Box(Modifier.weight(1f)) {
                    Bar(total.millis.toFloat() / longest, total.activity.color)
                }
                Text(
                    formatDuration(total.millis),
                    color = Chalk,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(start = 12.dp).width(64.dp),
                )
            }
        }
    }
}

@Composable
private fun Bar(fraction: Float, color: Color) {
    Box(Modifier.fillMaxWidth().height(8.dp).background(Track)) {
        Box(Modifier.fillMaxWidth(fraction.coerceIn(0f, 1f)).fillMaxHeight().background(color))
    }
}

fun formatDuration(millis: Long): String {
    val minutes = millis / 60_000
    return if (minutes < 60) "${minutes}m" else "${minutes / 60}h ${minutes % 60}m"
}
