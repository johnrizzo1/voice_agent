#!/usr/bin/env python3
"""
Test Bark TTS with the voice agent configuration.
"""

import asyncio
from voice_agent.core.config import Config
from voice_agent.core.tts_service import TTSService
from pathlib import Path

async def test_bark_tts():
    print("=== Testing Bark TTS ===\n")
    
    # Load configuration
    config_path = Path("voice_agent/config/default.yaml")
    config = Config.load(config_path)
    
    print(f"Config TTS engine: {config.tts.engine}")
    
    # Create TTS service
    tts_service = TTSService(config.tts)
    
    print(f"Selected backend: {tts_service.current_backend}")
    
    # Test backend detection
    if tts_service.current_backend != "bark":
        print("⚠️  Warning: Bark backend not selected, checking availability...")
        # Check imports
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            print("✅ Bark imports successful")
        except ImportError as e:
            print(f"❌ Bark import failed: {e}")
            return
    
    # Initialize service
    print("\nInitializing Bark TTS (this may take a while for model download)...")
    try:
        await tts_service.initialize()
        print(f"✅ Service initialized: {tts_service.is_initialized}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test synthesis
    print("\nTesting high-quality neural speech synthesis...")
    test_text = "Hello! This is a test of the Bark neural voice agent. The speech should sound very natural and human-like, much better than robotic synthetic voices."
    
    try:
        print("🎤 Generating speech (this may take 10-30 seconds)...")
        await tts_service.speak(test_text)
        print("✅ Speech synthesis completed successfully!")
    except Exception as e:
        print(f"❌ Speech synthesis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    await tts_service.cleanup()
    print("🧹 TTS service cleaned up")

if __name__ == "__main__":
    asyncio.run(test_bark_tts())