package io.github.matejpopovski.stridesense

import android.app.Application
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.github.matejpopovski.stridesense.core.Bout
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.data.BoutRepository
import io.github.matejpopovski.stridesense.data.StrideDatabase
import kotlin.test.assertEquals
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BoutStoreTest {
    private val database = Room
        .inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext<Application>(),
            StrideDatabase::class.java,
        )
        .allowMainThreadQueries()
        .build()
    private val repository = BoutRepository(database.bouts())

    @After
    fun closeDatabase() = database.close()

    private fun bout(start: Long, end: Long, activity: HumanActivity = HumanActivity.WALKING) =
        Bout(activity, start, end, windows = 1, confidenceSum = 0.9)

    @Test
    fun `saving a bout that grew rewrites its row`() = runBlocking {
        repository.save(bout(1_000, 2_000))
        repository.save(bout(1_000, 5_000))
        assertEquals(listOf(bout(1_000, 5_000)), repository.overlapping(0, 10_000).first())
    }

    @Test
    fun `an activity survives the round trip through storage`() = runBlocking {
        val stored = bout(1_000, 2_000, HumanActivity.WALKING_DOWNSTAIRS)
        repository.save(stored)
        assertEquals(stored, repository.overlapping(0, 10_000).first().single())
    }

    @Test
    fun `a day returns only bouts that overlap it, oldest first`() = runBlocking {
        repository.save(bout(500, 900))
        repository.save(bout(2_500, 3_000))
        repository.save(bout(1_500, 2_500))
        repository.save(bout(3_500, 4_000))
        val day = repository.overlapping(1_000, 3_000).first()
        assertEquals(listOf(1_500L, 2_500L), day.map { it.startMillis })
    }

    @Test
    fun `a bout crossing the start of the day belongs to that day`() = runBlocking {
        repository.save(bout(500, 1_500))
        assertEquals(1, repository.overlapping(1_000, 2_000).first().size)
    }
}
