package io.github.matejpopovski.stridesense

import androidx.test.ext.junit.runners.AndroidJUnit4
import io.github.matejpopovski.stridesense.core.Bout
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.data.BoutDao
import io.github.matejpopovski.stridesense.data.BoutEntity
import io.github.matejpopovski.stridesense.data.BoutRepository
import io.github.matejpopovski.stridesense.ui.DayRange
import io.github.matejpopovski.stridesense.ui.TodayViewModel
import io.github.matejpopovski.stridesense.ui.todayState
import java.time.ZoneId
import java.time.ZoneOffset
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

private const val DAY_MILLIS = 86_400_000L

private class RecordingDao : BoutDao {
    val rows = MutableStateFlow<List<BoutEntity>>(emptyList())
    val queriedDays = mutableListOf<Long>()

    override suspend fun upsert(bout: BoutEntity) {
        rows.value = rows.value + bout
    }

    override fun overlapping(dayStart: Long, dayEnd: Long): Flow<List<BoutEntity>> {
        queriedDays += dayStart
        return rows.map { list -> list.filter { it.endMillis > dayStart && it.startMillis < dayEnd } }
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(AndroidJUnit4::class)
class TodayStateTest {
    private val dispatcher = StandardTestDispatcher()

    @Before
    fun useTestMain() = Dispatchers.setMain(dispatcher)

    @After
    fun restoreMain() = Dispatchers.resetMain()

    private val day = DayRange(0, DAY_MILLIS)

    @Test
    fun `day range follows the local calendar, not UTC midnight`() {
        val tokyo = ZoneId.of("Asia/Tokyo")
        val range = DayRange.containing(0, tokyo)
        assertEquals(-9 * 3_600_000L, range.startMillis)
        assertEquals(15 * 3_600_000L, range.endMillis)
    }

    @Test
    fun `a day with a clock change is not assumed to be 24 hours`() {
        val newYork = ZoneId.of("America/New_York")
        val springForward = java.time.LocalDate.of(2026, 3, 8)
            .atStartOfDay(newYork).toInstant().toEpochMilli()
        val range = DayRange.containing(springForward + 1_000, newYork)
        assertEquals(23 * 3_600_000L, range.endMillis - range.startMillis)
    }

    @Test
    fun `timeline fractions place a bout within the day`() {
        val bout = Bout(HumanActivity.SITTING, DAY_MILLIS / 4, DAY_MILLIS / 2, 1, 1.0)
        val segment = todayState(LiveStatus(), listOf(bout), day).timeline.single()
        assertEquals(0.25f, segment.startFraction, 1e-6f)
        assertEquals(0.5f, segment.endFraction, 1e-6f)
    }

    @Test
    fun `a bout from yesterday adds only its part of today`() {
        val bout = Bout(HumanActivity.LAYING, -1_000, 1_000, 1, 1.0)
        val totals = todayState(LiveStatus(), listOf(bout), day).totals
        assertEquals(1_000, totals.single { it.activity == HumanActivity.LAYING }.millis)
    }

    @Test
    fun `every activity is listed even with nothing logged`() {
        val state = todayState(LiveStatus(), emptyList(), day)
        assertEquals(HumanActivity.entries, state.totals.map { it.activity })
        assertTrue(state.totals.all { it.millis == 0L })
        assertNull(state.activity)
    }

    private fun TestScope.collectState(model: TodayViewModel) {
        backgroundScope.launch(dispatcher) { model.state.collect {} }
    }

    @Test
    fun `live status and stored bouts combine into one state`() = runTest(dispatcher) {
        val dao = RecordingDao()
        val status = MutableStateFlow(LiveStatus())
        val model = TodayViewModel(BoutRepository(dao), status, { 1_000L }, ZoneOffset.UTC)
        collectState(model)
        status.value = LiveStatus(running = true, activity = HumanActivity.WALKING, confidence = 0.9f)
        dao.upsert(BoutEntity(0, 60_000, HumanActivity.WALKING.name, 40, 36.0))
        runCurrent()
        val state = model.state.value
        assertEquals(HumanActivity.WALKING, state.activity)
        assertEquals(60_000, state.totals.single { it.activity == HumanActivity.WALKING }.millis)
    }

    @Test
    fun `the screen moves on to the new day at midnight`() = runTest(dispatcher) {
        val dao = RecordingDao()
        val model = TodayViewModel(
            BoutRepository(dao),
            MutableStateFlow(LiveStatus()),
            { testScheduler.currentTime },
            ZoneOffset.UTC,
        )
        collectState(model)
        runCurrent()
        advanceTimeBy(DAY_MILLIS + 1)
        runCurrent()
        assertEquals(listOf(0L, DAY_MILLIS), dao.queriedDays)
    }
}
