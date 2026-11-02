// A complete build of its own, so `./gradlew -p core test` never applies the Android plugin and
// works on a machine with no Android SDK. The root build pulls it in with includeBuild.
pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
    versionCatalogs {
        create("libs") {
            from(files("../gradle/libs.versions.toml"))
        }
    }
}

rootProject.name = "core"
