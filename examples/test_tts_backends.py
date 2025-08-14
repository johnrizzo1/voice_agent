#!/usr/bin/env python3
"""
Test TTS backend availability and selection.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=== TTS Backend Availability Test ===\n")

# Test pyttsx3
print("1. Testing pyttsx3...")
try:
    import pyttsx3
    print("✅ pyttsx3 is available")
    pyttsx3_available = True
except ImportError as e:
    print(f"❌ pyttsx3 not available: {e}")
    pyttsx3_available = False

# Test Coqui TTS
print("\n2. Testing Coqui TTS...")
try:
    from TTS.api import TTS
    import torch
    print("✅ Coqui TTS is available")
    coqui_available = True
    
    # Try to initialize a model
    print("   Testing model initialization...")
    try:
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        print("✅ Coqui TTS model loaded successfully")
    except Exception as e:
        print(f"⚠️  Coqui TTS model loading failed: {e}")
        
except ImportError as e:
    print(f"❌ Coqui TTS not available: {e}")
    coqui_available = False
except Exception as e:
    print(f"❌ Coqui TTS error: {e}")
    coqui_available = False

# Test eSpeak-NG
print("\n3. Testing eSpeak-NG...")
try:
    import subprocess
    import shutil
    
    if shutil.which("espeak-ng"):
        print("✅ eSpeak-NG executable found")
        
        # Test version
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ eSpeak-NG version: {result.stdout.strip()}")
            espeak_available = True
        else:
            print(f"⚠️  eSpeak-NG version check failed: {result.stderr}")
            espeak_available = False
    else:
        print("❌ eSpeak-NG executable not found")
        espeak_available = False
        
except Exception as e:
    print(f"❌ eSpeak-NG test failed: {e}")
    espeak_available = False

# Test config loading
print("\n4. Testing configuration...")
try:
    from voice_agent.core.config import Config
    from pathlib import Path
    
    config_path = Path("voice_agent/config/default.yaml")
    if config_path.exists():
        config = Config.load(config_path)
        print(f"✅ Config loaded - TTS engine: {config.tts.engine}")
    else:
        print("❌ Config file not found")
        
except Exception as e:
    print(f"❌ Config loading failed: {e}")

# Test TTS service
print("\n5. Testing TTS Service...")
try:
    from voice_agent.core.config import TTSConfig
    from voice_agent.core.tts_service import TTSService
    
    tts_config = TTSConfig(engine="coqui")  # Force coqui engine
    tts_service = TTSService(tts_config)
    
    print(f"✅ TTS Service created")
    print(f"   Current backend: {tts_service.current_backend}")
    print(f"   Config engine: {tts_service.config.engine}")
    
except Exception as e:
    print(f"❌ TTS Service test failed: {e}")

print(f"\n=== Summary ===")
print(f"pyttsx3: {'✅' if pyttsx3_available else '❌'}")
print(f"Coqui TTS: {'✅' if coqui_available else '❌'}")
print(f"eSpeak-NG: {'✅' if espeak_available else '❌'}")