[app]

# (str) Title of your application
title = Federated Learning Client

# (str) Package name
package.name = flclient

# (str) Package domain (needed for android/ios packaging)
package.domain = org.flclient

# (str) Source files where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,json,tflite

# (list) List of inclusions using pattern matching
#source.include_patterns = assets/*,images/*.png

# (list) Source files to exclude (let empty to not exclude anything)
#source.exclude_exts = spec

# (list) List of directory to exclude (let empty to not exclude anything)
#source.exclude_dirs = tests, bin, venv

# (list) List of exclusions using pattern matching
#source.exclude_patterns = license,images/*/*.jpg

# (str) Application versioning (method 1)
version = 0.1

# (str) Application versioning (method 2)
# version.regex = __version__ = ['"](.*)['"]
# version.filename = %(source.dir)s/main.py

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3,kivy,flwr,numpy,psutil,pyjnius,android

# (str) Custom source folders for requirements
# Sets custom source for any requirements with recipes
# requirements.source.kivy = ../../kivy

# (list) Presets of requirements to avoid duplicate requirements
# (str) Preset of requirements for Android
android.presets = python3

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.arch = arm64-v8a

# (int) minimum API level required
android.minapi = 21

# (int) Android NDK API level to use
android.ndk = 21e

# (int) Android SDK API level to use
android.sdk = 33

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
#android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
#android.sdk_path =

# (str) ANT directory (if empty, it will be automatically downloaded.)
#android.ant_path =

# (bool) If True, then skip trying to update the Android sdk
# This can be useful to avoid excess Internet downloads or save time
# when an update is due and you just want to test/build your package
# android.skip_update = False

# (bool) If True, then automatically accept SDK license
# agreements. This is intended for automation only. If set to False,
# the default, you will be asked to accept the license agreements.
# android.accept_sdk_license = False

# (str) Android entry point, default is ok for Kivy-based app
#android.entrypoint = org.kivy.android.PythonActivity

# (list) Pattern to whitelist for the whole project
#android.whitelist =

# (str) Path to a custom whitelist file
#android.whitelist_src =

# (str) Path to a custom blacklist file
#android.blacklist_src =

# (list) List of Android .gradle dependencies to add (using guava syntax as a string)
# This will allow you to use the latest version available
#android.gradle_dependencies =

# (list) Android add-on libraries to include (comma separated)
android.add_libs = python3

# (list) Java classes to add as activities to the manifest.
#android.add_activities =

# (str) OUYA Console category. Should be one of GAME, APP or SHELL.
# ouya.category = GAME

# (str) Filename of OUYA Console icon. It must be a 732x412 png image.
# ouya.icon.filename = %(source.dir)s/data/ouya_icon.png

# (str) XML file to include as an intent filters in <activity> tag
#android.manifest.intent_filters =

# (str) launchMode to set for the main activity
#android.manifest.launch_mode = standard

# (list) Android additional libraries to copy into libs/armeabi
#android.add_libs_armeabi = libs/android/*.so
#android.add_libs_armeabi_v7a = libs/android-v7/*.so
#android.add_libs_arm64_v8a = libs/android-v8/*.so
#android.add_libs_x86 = libs/android-x86/*.so
#android.add_libs_x86_64 = libs/android-x86_64/*.so

# (bool) Indicate whether the screen should stay on
# Don't work with all videos and Android versions
android.wakelock = False

# (list) Android application meta-data to set (key=value format)
#android.meta_data =

# (list) Android library project to add (will be added in the
# project.properties automatically.)
#android.library_references =

# (list) Android shared libraries to add to the project (will be added in project.properties)
#android.libraries =

# (str) Android logcat filters to use
android.logcat_filters = *:S python:D

# (bool) Copy library instead of making a libpymodules.so
#android.copy_libs = 1

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
# In past, was `android.arch` as we weren't supporting builds for multiple archs at the same time.
android.archs = arm64-v8a

# (int) overrides automatic version code computation for version checks
# android.numeric_version = 1

# (bool) enables Android auto backup feature (Android API >=23)
android.allow_backup = True

# (str) XML file for custom backup rules (see official Android documentation)
#android.backup_rules =

# (str) If you need to insert custom attributes in the <application> tag of the manifest
#android.manifest_placeholders = [:]

# (str) Custom launcher icon (relative to source.dir)
# android.icon = assets/icon.png

# (str) Custom splash screen (relative to source.dir)
android.presplash_color = #FFFFFF

# (str) Supported orientation (one of 'landscape', 'portrait' or 'sensor')
android.orientation = portrait

# (list) Android permission
android.permissions = INTERNET,ACCESS_NETWORK_STATE,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1
