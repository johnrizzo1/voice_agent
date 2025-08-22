#!/usr/bin/env python3
"""
Test script to verify SoundDevice migration works correctly.
This script tests the new AudioManager implementation.
"""

import asyncio
import sys
import os

# Add src to path to import voice_agent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from voice_agent.core.audio_manager import AudioManager
from voice_agent.core.config import AudioConfig


def test_sounddevice_import():
    """Test that SoundDevice can be imported and basic functions work."""
    try:
        import sounddevice as sd

        print(f"✅ SoundDevice import successful - version: {sd.__version__}")

        # Test device enumeration
        devices = sd.query_devices()
        print(f"✅ Found {len(devices)} audio devices")

        # Show default devices
        default_input, default_output = sd.default.device
        print(f"✅ Default input device: {default_input}")
        print(f"✅ Default output device: {default_output}")

        return True
    except Exception as e:
        print(f"❌ SoundDevice import failed: {e}")
        return False


def test_audio_config():
    """Test that AudioConfig can be created with new SoundDevice settings."""
    try:
        # Test creating config with new SoundDevice-specific settings
        config_data = {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "input_device": None,
            "output_device": None,
            "channels": 1,
            "dtype": "int16",
            "latency": "low",
            "buffer_size": 2048,
            "enable_echo_cancellation": True,
            "vad_aggressiveness": 3,
            "energy_threshold": 7000,
            "barge_in_enabled": True,
            "barge_in_energy_threshold": 0.28,
            "barge_in_consecutive_frames": 5,
            "feedback_prevention_enabled": True,
            "buffer_clear_on_playback": True,
        }

        # Create AudioConfig (assuming it accepts dict or has from_dict method)
        config = AudioConfig(**config_data)
        print("✅ AudioConfig created successfully with SoundDevice settings")
        print(f"   - Sample rate: {config.sample_rate}")
        print(f"   - Channels: {getattr(config, 'channels', 'not set')}")
        print(f"   - Data type: {getattr(config, 'dtype', 'not set')}")
        print(f"   - Latency: {getattr(config, 'latency', 'not set')}")

        return config
    except Exception as e:
        print(f"❌ AudioConfig creation failed: {e}")
        return None


async def test_audio_manager_init(config):
    """Test that AudioManager can be initialized with SoundDevice."""
    try:
        # Create AudioManager instance
        audio_manager = AudioManager(config)
        print("✅ AudioManager created successfully")

        # Test initialization
        await audio_manager.initialize()
        print("✅ AudioManager initialized successfully")

        # Test device info
        device_info = audio_manager.get_device_info()
        print(
            f"✅ Device info retrieved - Input devices: {len(device_info.get('input_devices', []))}, Output devices: {len(device_info.get('output_devices', []))}"
        )

        # Test status
        status = audio_manager.get_status()
        print("✅ Status retrieved successfully")
        print(f"   - Input available: {status.get('input_available', False)}")
        print(f"   - Microphone error: {status.get('microphone_error', False)}")

        # Cleanup
        await audio_manager.cleanup()
        print("✅ AudioManager cleanup successful")

        return True
    except Exception as e:
        print(f"❌ AudioManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_migration_completeness():
    """Test that ALSA-specific code has been removed."""
    try:
        # Read the audio_manager.py file and check for ALSA remnants
        with open("src/voice_agent/core/audio_manager.py", "r") as f:
            content = f.read()

        alsa_indicators = [
            "_suppress_alsa_jack_errors",
            "_py_alsa_error_handler",
            "libasound.so",
            "ctypes.cdll.LoadLibrary",
            "snd_lib_error_set_handler",
            "pyaudio.PyAudio",
            "import pyaudio",
        ]

        found_alsa = []
        for indicator in alsa_indicators:
            if indicator in content:
                found_alsa.append(indicator)

        if found_alsa:
            print(f"❌ ALSA remnants found: {found_alsa}")
            return False
        else:
            print("✅ No ALSA-specific code found - migration appears complete")
            return True

    except Exception as e:
        print(f"❌ Migration completeness check failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🔧 Testing SoundDevice Migration")
    print("=" * 50)

    # Test 1: SoundDevice import
    print("\n1. Testing SoundDevice Import...")
    if not test_sounddevice_import():
        print("❌ Critical failure - SoundDevice import failed")
        return False

    # Test 2: AudioConfig
    print("\n2. Testing AudioConfig...")
    config = test_audio_config()
    if config is None:
        print("❌ Critical failure - AudioConfig creation failed")
        return False

    # Test 3: AudioManager
    print("\n3. Testing AudioManager...")
    if not await test_audio_manager_init(config):
        print("❌ AudioManager test failed")
        return False

    # Test 4: Migration completeness
    print("\n4. Testing Migration Completeness...")
    if not test_migration_completeness():
        print("❌ Migration completeness check failed")
        return False

    print("\n" + "=" * 50)
    print("🎉 All tests passed! SoundDevice migration appears successful.")
    print("\nNext steps:")
    print("1. Test with actual voice agent application")
    print("2. Verify cross-platform compatibility")
    print("3. Performance testing and optimization")

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
