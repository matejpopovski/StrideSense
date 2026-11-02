package io.github.matejpopovski.stridesense.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.github.matejpopovski.stridesense.LiveStatus
import io.github.matejpopovski.stridesense.SensingProblem
import io.github.matejpopovski.stridesense.core.Bout
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.core.clippedTo
import io.github.matejpopovski.stridesense.core.dailyTotals
import io.github.matejpopovski.stridesense.data.BoutRepository
import java.time.Instant
import java.time.ZoneId
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.stateIn

data class TimelineSegment(val activity: HumanActivity, val startFraction: Float, val endFraction: Float)

data class ActivityTotal(val activity: HumanActivity, val millis: Long)

data class TodayUiState(
    val running: Boolean = false,
    val activity: HumanActivity? = null,
    val confidence: Float = 0f,
    val problem: SensingProblem? = null,
    val timeline: List<TimelineSegment> = emptyList(),
    val totals: List<ActivityTotal> = HumanActivity.entries.map { ActivityTotal(it, 0L) },
)

/** A local calendar day as a half-open millisecond range. */
data class DayRange(val startMillis: Long, val endMillis: Long) {
    companion object {
        fun containing(millis: Long, zone: ZoneId): DayRange {
            val date = Instant.ofEpochMilli(millis).atZone(zone).toLocalDate()
            return DayRange(
                date.atStartOfDay(zone).toInstant().toEpochMilli(),
                date.plusDays(1).atStartOfDay(zone).toInstant().toEpochMilli(),
            )
        }
    }
}

fun todayState(status: LiveStatus, bouts: List<Bout>, day: DayRange): TodayUiState {
    val length = (day.endMillis - day.startMillis).toFloat()
    val timeline = bouts.mapNotNull { bout ->
        bout.clippedTo(day.startMillis, day.endMillis)?.let {
            TimelineSegment(
                it.activity,
                (it.startMillis - day.startMillis) / length,
                (it.endMillis - day.startMillis) / length,
            )
        }
    }
    val totals = dailyTotals(bouts, day.startMillis, day.endMillis)
    return TodayUiState(
        running = status.running,
        activity = status.activity,
        confidence = status.confidence,
        problem = status.problem,
        timeline = timeline,
        totals = HumanActivity.entries.map { ActivityTotal(it, totals.getValue(it)) },
    )
}

@OptIn(ExperimentalCoroutinesApi::class)
class TodayViewModel(
    bouts: BoutRepository,
    status: StateFlow<LiveStatus>,
    clockMillis: () -> Long = System::currentTimeMillis,
    zone: ZoneId = ZoneId.systemDefault(),
) : ViewModel() {
    private val days: Flow<DayRange> = flow {
        while (true) {
            val day = DayRange.containing(clockMillis(), zone)
            emit(day)
            delay(day.endMillis - clockMillis())
        }
    }

    val state: StateFlow<TodayUiState> = days
        .flatMapLatest { day ->
            combine(status, bouts.overlapping(day.startMillis, day.endMillis)) { live, stored ->
                todayState(live, stored, day)
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(STOP_TIMEOUT_MILLIS), TodayUiState())

    private companion object {
        const val STOP_TIMEOUT_MILLIS = 5_000L
    }
}
