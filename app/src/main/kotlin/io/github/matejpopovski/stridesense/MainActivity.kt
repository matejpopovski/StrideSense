package io.github.matejpopovski.stridesense

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import io.github.matejpopovski.stridesense.sensing.ActivitySensorService
import io.github.matejpopovski.stridesense.ui.TodayScreen
import io.github.matejpopovski.stridesense.ui.TodayViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as StrideApplication
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                val model: TodayViewModel = viewModel(
                    factory = viewModelFactory {
                        initializer { TodayViewModel(app.bouts, app.liveStatus) }
                    },
                )
                val state by model.state.collectAsStateWithLifecycle()
                val context = LocalContext.current
                // Logging works without the notification permission, so a refusal changes nothing.
                val notificationPermission = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { ActivitySensorService.start(context) }

                TodayScreen(
                    state = state,
                    onToggle = {
                        val needsAsking = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                            ContextCompat.checkSelfPermission(
                                context,
                                Manifest.permission.POST_NOTIFICATIONS,
                            ) != PackageManager.PERMISSION_GRANTED
                        when {
                            state.running -> ActivitySensorService.stop(context)

                            needsAsking ->
                                notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)

                            else -> ActivitySensorService.start(context)
                        }
                    },
                )
            }
        }
    }
}
