import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

group = "io.github.matejpopovski.stridesense"
version = "0.1.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
        allWarningsAsErrors = true
    }
}

val ktlint = configurations.create("ktlint")

dependencies {
    ktlint(libs.ktlint.cli)
    implementation(libs.kotlinx.serialization.json)
    testImplementation(libs.kotlin.test)
    testImplementation(libs.junit)
}

tasks.test {
    useJUnit()
    systemProperty("stridesense.root", layout.projectDirectory.dir("..").asFile.absolutePath)
    testLogging {
        events("failed")
        showStandardStreams = false
    }
}

// Lives in this build rather than the root one so it can reach the app's sources too without
// configuring the Android plugin.
val lintedSources = listOf("src/**/*.kt", "*.kts", "../app/src/**/*.kt", "../app/*.kts", "../*.kts")

tasks.register<JavaExec>("ktlintCheck") {
    description = "Checks the style of all Kotlin in the repository."
    classpath = ktlint
    mainClass = "com.pinterest.ktlint.Main"
    args = lintedSources
    jvmArgs("--add-opens=java.base/java.lang=ALL-UNNAMED")
}

tasks.register<JavaExec>("ktlintFormat") {
    description = "Rewrites all Kotlin in the repository to match the style."
    classpath = ktlint
    mainClass = "com.pinterest.ktlint.Main"
    args = listOf("--format") + lintedSources
    jvmArgs("--add-opens=java.base/java.lang=ALL-UNNAMED")
}
