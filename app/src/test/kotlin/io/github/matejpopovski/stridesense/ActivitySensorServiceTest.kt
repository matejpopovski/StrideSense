package io.github.matejpopovski.stridesense

import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorManager
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.sensing.ActivitySensorService
import io.github.matejpopovski.stridesense.sensing.ClosableClassifier
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.Shadows.shadowOf
import org.robolectric.android.controller.ServiceController
import org.robolectric.shadows.SensorEventBuilder
import org.robolectric.shadows.ShadowSensor

/** Stands in for LiteRT, whose native library is built for Android and cannot load on a JVM. */
private class FakeClassifier(private val best: HumanActivity) : ClosableClassifier {
    val closed = CountDownLatch(1)
    var calls = 0

    override fun classify(window: FloatArray): FloatArray {
        calls++
        return FloatArray(HumanActivity.entries.size).also { it[best.ordinal] = 1f }
    }

    override fun close() = closed.countDown()
}

@RunWith(AndroidJUnit4::class)
class ActivitySensorServiceTest {
    private val app = ApplicationProvider.getApplicationContext<StrideApplication>()
    private val sensorManager = app.getSystemService(SensorManager::class.java)
    private val shadowManager = shadowOf(sensorManager)
    private val classifier = FakeClassifier(HumanActivity.WALKING)

    private fun addSensors() {
        shadowManager.addSensor(ShadowSensor.newInstance(Sensor.TYPE_ACCELEROMETER))
        shadowManager.addSensor(ShadowSensor.newInstance(Sensor.TYPE_GYROSCOPE))
    }

    private fun started(intent: Intent? = null): ServiceController<ActivitySensorService> =
        Robolectric.buildService(ActivitySensorService::class.java, intent).create().startCommand(0, 1)

    private fun send(type: Int, sample: Int) {
        val gravity = if (type == Sensor.TYPE_ACCELEROMETER) 9.8f else 0f
        val event = SensorEventBuilder.newBuilder()
            .setSensor(ShadowSensor.newInstance(type))
            .setTimestamp(sample * 20_000_000L)
            .setValues(floatArrayOf(0f, 0f, gravity))
            .build()
        shadowManager.sendSensorEventToListeners(event)
    }

    @Before
    fun useFakeClassifier() {
        app.classifierFactory = { classifier }
        app.liveStatus.value = LiveStatus()
    }

    @Test
    fun `starting goes foreground with the health type and listens to both sensors`() {
        addSensors()
        val service = started().get()
        val shadow = shadowOf(service)
        assertNotNull(shadow.lastForegroundNotification)
        assertEquals(ActivitySensorService.NOTIFICATION_ID, shadow.lastForegroundNotificationId)
        assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH, service.foregroundServiceType)
        assertTrue(
            shadowManager.hasListener(service, sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)),
        )
        assertTrue(shadowManager.hasListener(service, sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)))
        assertTrue(app.liveStatus.value.running)
    }

    @Test
    fun `a filled window reaches the classifier and the live status`() {
        addSensors()
        started()
        for (sample in 0 until 128) {
            send(Sensor.TYPE_ACCELEROMETER, sample)
            send(Sensor.TYPE_GYROSCOPE, sample)
        }
        assertEquals(1, classifier.calls)
        assertEquals(HumanActivity.WALKING, app.liveStatus.value.activity)
    }

    @Test
    fun `the stop action stops the service`() {
        addSensors()
        val intent = Intent(app, ActivitySensorService::class.java)
            .setAction(ActivitySensorService.ACTION_STOP)
        val service = started(intent).get()
        assertTrue(shadowOf(service).isStoppedBySelf)
    }

    @Test
    fun `destroying releases the sensors and closes the classifier`() {
        addSensors()
        val controller = started()
        controller.destroy()
        assertTrue(shadowManager.listeners.isEmpty())
        assertTrue(classifier.closed.await(5, TimeUnit.SECONDS))
        assertFalse(app.liveStatus.value.running)
    }

    @Test
    fun `a device without a gyroscope reports the problem instead of crashing`() {
        shadowManager.addSensor(ShadowSensor.newInstance(Sensor.TYPE_ACCELEROMETER))
        val controller = started()
        assertTrue(shadowOf(controller.get()).isStoppedBySelf)
        controller.destroy()
        assertEquals(SensingProblem.NO_SENSORS, app.liveStatus.value.problem)
    }

    @Test
    fun `a sensor that refuses to register reports the problem`() {
        addSensors()
        shadowManager.setForceListenersToFail(true)
        val controller = started()
        assertTrue(shadowOf(controller.get()).isStoppedBySelf)
        controller.destroy()
        assertEquals(SensingProblem.SENSOR_REFUSED, app.liveStatus.value.problem)
    }
}
