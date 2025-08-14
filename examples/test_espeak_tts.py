#!/usr/bin/env python3
"""
Test eSpeak-NG TTS with the voice agent configuration.
"""

import asyncio
from voice_agent.core.config import Config
from voice_agent.core.tts_service import TTSService
from pathlib import Path

async def test_espeak_tts():
    print("=== Testing eSpeak-NG TTS ===\n")
    
    # Load configuration
    config_path = Path("voice_agent/config/default.yaml")
    config = Config.load(config_path)
    
    print(f"Config TTS engine: {config.tts.engine}")
    
    # Create TTS service
    tts_service = TTSService(config.tts)
    
    print(f"Selected backend: {tts_service.current_backend}")
    
    # Initialize service
    await tts_service.initialize()
    
    print(f"Service initialized: {tts_service.is_initialized}")
    
    # Test synthesis
    print("\nTesting speech synthesis...")
    test_text = "Hello! This is a test of the enhanced voice agent with eSpeak-NG TTS. The voice should sound much more natural than the robotic pyttsx3 voice."
    
    try:
        await tts_service.speak(test_text)
        print("✅ Speech synthesis completed successfully!")
    except Exception as e:
        print(f"❌ Speech synthesis failed: {e}")
    
    # Cleanup
    await tts_service.cleanup()
    print("🧹 TTS service cleaned up")

if __name__ == "__main__":
    asyncio.run(test_espeak_tts())