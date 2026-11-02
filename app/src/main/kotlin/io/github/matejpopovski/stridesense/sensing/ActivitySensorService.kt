package io.github.matejpopovski.stridesense.sensing

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import io.github.matejpopovski.stridesense.LiveStatus
import io.github.matejpopovski.stridesense.R
import io.github.matejpopovski.stridesense.SensingProblem
import io.github.matejpopovski.stridesense.StrideApplication
import io.github.matejpopovski.stridesense.core.Bout
import io.github.matejpopovski.stridesense.core.HumanActivity
import io.github.matejpopovski.stridesense.core.LivePrediction
import io.github.matejpopovski.stridesense.core.SensingSession
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.launch

/**
 * Keeps sampling while the app is in the background. Everything that touches the pipeline or the
 * interpreter runs on [sensorThread], so neither needs a lock and inference never blocks the UI.
 */
class ActivitySensorService :
    Service(),
    SensorEventListener {
    private val app get() = application as StrideApplication
    private var sensorThread: HandlerThread? = null
    private var session: SensingSession? = null
    private var classifier: ClosableClassifier? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var shownActivity: HumanActivity? = null
    private var failure: SensingProblem? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (sensorThread == null) startSensing()
        return START_NOT_STICKY
    }

    private fun startSensing() {
        createNotificationChannel()
        // Android requires this promptly after startForegroundService, even if sensing then fails.
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            notification(getString(R.string.notification_waiting)),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH,
        )

        val sensors = getSystemService<SensorManager>()
        val accelerometer = sensors?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensors?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        if (sensors == null || accelerometer == null || gyroscope == null) {
            failWith(SensingProblem.NO_SENSORS)
            return
        }

        val thread = HandlerThread("stridesense-sensors").apply { start() }
        sensorThread = thread
        val handler = Handler(thread.looper)
        wakeLock = getSystemService<PowerManager>()
            ?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "stridesense:sensing")
            ?.apply { acquire(TimeUnit.HOURS.toMillis(WAKE_LOCK_HOURS)) }

        val classifier = app.classifierFactory().also { classifier = it }
        session = SensingSession(
            spec = app.spec,
            classifier = classifier,
            clockMillis = System::currentTimeMillis,
            onBout = ::save,
            onPrediction = ::publish,
        )
        app.liveStatus.value = LiveStatus(running = true)

        val accepted =
            sensors.registerListener(this, accelerometer, SAMPLING_PERIOD_US, handler) &&
                sensors.registerListener(this, gyroscope, SAMPLING_PERIOD_US, handler)
        if (!accepted) failWith(SensingProblem.SENSOR_REFUSED)
    }

    private fun failWith(problem: SensingProblem) {
        failure = problem
        stopSelf()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val active = session ?: return
        val (x, y, z) = event.values
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> active.onAccelerometer(event.timestamp, x, y, z)
            Sensor.TYPE_GYROSCOPE -> active.onGyroscope(event.timestamp, x, y, z)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun save(bout: Bout) {
        val application = app
        application.writeScope.launch { application.bouts.save(bout) }
    }

    private fun publish(prediction: LivePrediction) {
        app.liveStatus.value = LiveStatus(
            running = true,
            activity = prediction.activity,
            confidence = prediction.confidence,
        )
        if (prediction.activity != shownActivity) {
            shownActivity = prediction.activity
            getSystemService<NotificationManager>()?.notify(
                NOTIFICATION_ID,
                notification(prediction.activity.displayName),
            )
        }
    }

    override fun onDestroy() {
        getSystemService<SensorManager>()?.unregisterListener(this)
        session = null
        val thread = sensorThread
        val heldClassifier = classifier
        if (thread != null) {
            // Queued sensor events finish first, so the interpreter is never closed mid-inference.
            Handler(thread.looper).post { heldClassifier?.close() }
            thread.quitSafely()
        }
        sensorThread = null
        classifier = null
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
        app.liveStatus.value = LiveStatus(running = false, problem = failure)
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel),
            NotificationManager.IMPORTANCE_LOW,
        )
        getSystemService<NotificationManager>()?.createNotificationChannel(channel)
    }

    private fun notification(text: String): Notification {
        val stopIntent = PendingIntent.getService(
            this,
            0,
            Intent(this, ActivitySensorService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(0, getString(R.string.notification_stop), stopIntent)
            .build()
    }

    companion object {
        const val NOTIFICATION_ID = 1
        const val ACTION_STOP = "io.github.matejpopovski.stridesense.action.STOP"
        private const val CHANNEL_ID = "activity-logging"

        /** 50 Hz, the rate of the training data. The OS treats this as a hint. */
        const val SAMPLING_PERIOD_US = 20_000
        private const val WAKE_LOCK_HOURS = 16L

        fun start(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, ActivitySensorService::class.java),
            )
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, ActivitySensorService::class.java))
        }
    }
}
