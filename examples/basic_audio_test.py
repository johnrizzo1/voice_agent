#!/usr/bin/env python3
"""
Basic audio pipeline test for the voice agent.

This test demonstrates the core audio pipeline working end-to-end:
1. Captures audio from microphone
2. Transcribes it using STT service
3. Speaks it back using TTS service

This is a simple echo test to validate the audio components.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the voice_agent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_agent.core.config import Config
from voice_agent.core.audio_manager import AudioManager
from voice_agent.core.stt_service import STTService
from voice_agent.core.tts_service import TTSService


async def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


async def test_audio_devices(audio_manager: AudioManager):
    """Test audio device availability."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing audio device availability...")
    
    device_info = audio_manager.get_device_info()
    
    logger.info(f"Default input device: {device_info.get('default_input', {}).get('name', 'None')}")
    logger.info(f"Default output device: {device_info.get('default_output', {}).get('name', 'None')}")
    
    input_devices = device_info.get('input_devices', [])
    output_devices = device_info.get('output_devices', [])
    
    logger.info(f"Available input devices: {len(input_devices)}")
    for device in input_devices[:3]:  # Show first 3
        logger.info(f"  - {device['name']} (channels: {device['channels']})")
    
    logger.info(f"Available output devices: {len(output_devices)}")
    for device in output_devices[:3]:  # Show first 3
        logger.info(f"  - {device['name']} (channels: {device['channels']})")


async def test_stt_models(stt_service: STTService):
    """Test STT model availability."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing STT models...")
    
    supported_models = stt_service.get_supported_models()
    logger.info(f"Supported STT models: {supported_models}")
    
    model_info = stt_service.get_model_info()
    logger.info(f"Current STT configuration: {model_info}")


async def test_tts_voices(tts_service: TTSService):
    """Test TTS voice availability."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing TTS voices...")
    
    service_info = tts_service.get_service_info()
    logger.info(f"Current TTS configuration: {service_info}")
    
    voices = tts_service.get_available_voices()
    logger.info(f"Available voices: {len(voices)}")
    for voice in voices[:3]:  # Show first 3
        logger.info(f"  - {voice['name']} ({voice.get('backend', 'unknown')})")


async def test_audio_pipeline(audio_manager: AudioManager, stt_service: STTService, tts_service: TTSService):
    """Test the complete audio pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("STARTING AUDIO PIPELINE TEST")
    logger.info("=" * 50)
    
    try:
        # Test 1: Simple TTS test
        logger.info("\n1. Testing TTS synthesis...")
        test_text = "Hello! This is a test of the text-to-speech system. Can you hear me?"
        await tts_service.speak(test_text)
        logger.info("TTS completed, ready for input...")
        
        # Test 2: Audio capture test
        logger.info("\n2. Testing audio capture...")
        logger.info("Please speak something (you have 10 seconds)...")
        
        # Start a timeout for recording
        try:
            audio_data = await asyncio.wait_for(
                audio_manager.listen(),
                timeout=10.0
            )
            
            if audio_data is not None:
                logger.info(f"Captured audio data: {len(audio_data)} samples")
                
                # Test 3: STT transcription
                logger.info("\n3. Testing speech-to-text...")
                transcribed_text = await stt_service.transcribe(audio_data)
                
                if transcribed_text:
                    logger.info(f"Transcribed text: '{transcribed_text}'")
                    
                    # Test 4: Complete pipeline (echo back)
                    logger.info("\n4. Testing complete pipeline (echo back)...")
                    echo_text = f"I heard you say: {transcribed_text}"
                    await tts_service.speak(echo_text)
                    
                    logger.info("✅ Audio pipeline test completed successfully!")
                    return True
                else:
                    logger.warning("No text was transcribed from the audio")
                    
            else:
                logger.warning("No audio data was captured")
                
        except asyncio.TimeoutError:
            logger.warning("Audio capture timed out - no speech detected")
            
    except Exception as e:
        logger.error(f"Audio pipeline test failed: {e}")
        return False
    
    logger.info("⚠️ Audio pipeline test completed with issues")
    return False


async def main():
    """Main test function."""
    logger = await setup_logging()
    
    try:
        logger.info("Starting basic audio pipeline test...")
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "voice_agent" / "config" / "default.yaml"
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return
        
        config = Config.load(config_path)
        
        # Use Coqui TTS for better audio control and feedback prevention
        config.tts.engine = "coqui"
        
        logger.info("Loaded configuration successfully")
        
        # Initialize components
        logger.info("\nInitializing audio components...")
        
        audio_manager = AudioManager(config.audio)
        await audio_manager.initialize()
        logger.info("✅ Audio manager initialized")
        
        stt_service = STTService(config.stt)
        await stt_service.initialize()
        logger.info("✅ STT service initialized")
        
        tts_service = TTSService(config.tts, audio_manager)
        await tts_service.initialize()
        logger.info("✅ TTS service initialized")
        
        # Run device tests
        await test_audio_devices(audio_manager)
        await test_stt_models(stt_service)
        await test_tts_voices(tts_service)
        
        # Run the main audio pipeline test
        success = await test_audio_pipeline(audio_manager, stt_service, tts_service)
        
        # Cleanup
        logger.info("\nCleaning up...")
        await audio_manager.cleanup()
        await stt_service.cleanup()
        await tts_service.cleanup()
        
        if success:
            logger.info("\n🎉 All tests passed! Audio pipeline is working correctly.")
        else:
            logger.info("\n⚠️ Some tests had issues. Check the logs above.")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Voice Agent - Basic Audio Pipeline Test")
    print("=" * 40)
    print("This test will:")
    print("1. Initialize all audio components")
    print("2. Test device availability")
    print("3. Test text-to-speech")
    print("4. Capture audio from microphone")
    print("5. Transcribe the audio")
    print("6. Speak back what was heard")
    print()
    print("Make sure you have a working microphone and speakers!")
    print("Press Ctrl+C to stop at any time.")
    print("=" * 40)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()