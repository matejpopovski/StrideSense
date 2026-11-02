package io.github.matejpopovski.stridesense

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.security.MessageDigest
import kotlin.test.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Robolectric cannot execute the model, so these check what it can: that the app packages the very
 * model and normalization the export step produced. Inference itself is checked against golden
 * outputs in Python and by LiteRtClassifierInstrumentedTest.
 */
@RunWith(AndroidJUnit4::class)
class PackagedModelTest {
    private val app = ApplicationProvider.getApplicationContext<StrideApplication>()
    private val model = app.assets.open(app.spec.modelFile).use { it.readBytes() }

    @Test
    fun `the packaged model is the one the spec was exported with`() {
        val digest = MessageDigest.getInstance("SHA-256").digest(model)
        val hex = digest.joinToString("") { "%02x".format(it) }
        assertEquals(app.spec.modelSha256, hex)
    }

    @Test
    fun `the packaged model is a LiteRT flatbuffer`() {
        assertEquals("TFL3", String(model, 4, 4, Charsets.US_ASCII))
    }

    @Test
    fun `the spec describes the window the model takes`() {
        assertEquals(128, app.spec.windowLength)
        assertEquals(64, app.spec.windowStride)
        assertEquals(20_000_000, app.spec.samplePeriodNs)
    }
}
