package io.github.matejpopovski.stridesense.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Upsert
import io.github.matejpopovski.stridesense.core.Bout
import io.github.matejpopovski.stridesense.core.HumanActivity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/** Keyed on the start time: a growing bout is rewritten in place on every window. */
@Entity(tableName = "bouts")
data class BoutEntity(
    @PrimaryKey val startMillis: Long,
    val endMillis: Long,
    val activity: String,
    val windows: Int,
    val confidenceSum: Double,
)

@Dao
interface BoutDao {
    @Upsert
    suspend fun upsert(bout: BoutEntity)

    @Query(
        "SELECT * FROM bouts WHERE endMillis > :dayStart AND startMillis < :dayEnd " +
            "ORDER BY startMillis",
    )
    fun overlapping(dayStart: Long, dayEnd: Long): Flow<List<BoutEntity>>
}

@Database(entities = [BoutEntity::class], version = 1, exportSchema = false)
abstract class StrideDatabase : RoomDatabase() {
    abstract fun bouts(): BoutDao

    companion object {
        fun open(context: Context): StrideDatabase =
            Room.databaseBuilder(context, StrideDatabase::class.java, "stridesense.db").build()
    }
}

class BoutRepository(private val dao: BoutDao) {
    suspend fun save(bout: Bout) = dao.upsert(bout.toEntity())

    fun overlapping(dayStartMillis: Long, dayEndMillis: Long): Flow<List<Bout>> =
        dao.overlapping(dayStartMillis, dayEndMillis).map { rows -> rows.map { it.toBout() } }
}

private fun Bout.toEntity() = BoutEntity(startMillis, endMillis, activity.name, windows, confidenceSum)

private fun BoutEntity.toBout() =
    Bout(enumValueOf<HumanActivity>(activity), startMillis, endMillis, windows, confidenceSum)
