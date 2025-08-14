"""
Audio input/output management for the voice agent.
"""

import asyncio
import logging
from typing import Callable, List, Optional

import numpy as np
import pyaudio
import webrtcvad

from .config import AudioConfig


class AudioManager:
    """
    Manages audio input and output for the voice agent.

    Handles:
    - Microphone input capture
    - Audio preprocessing and chunking
    - Voice activity detection
    - Audio output/playback
    - Real-time buffering
    """

    def __init__(self, config: AudioConfig):
        """
        Initialize the audio manager.

        Args:
            config: Audio configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # PyAudio components
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

        # Voice activity detection
        self.vad: Optional[webrtcvad.Vad] = None

        # Audio buffers
        self.input_buffer: List[bytes] = []
        self.is_recording = False
        self.is_speaking = False  # Flag to prevent feedback during TTS
        self.last_speech_end_time = 0.0  # Track when TTS last ended

        # Callbacks
        self.audio_callback: Optional[Callable] = None

    async def initialize(self) -> None:
        """Initialize audio components."""
        self.logger.info("Initializing audio manager...")

        try:
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            self.logger.info("PyAudio initialized")

            # Initialize VAD with lower aggressiveness for better sensitivity
            self.vad = webrtcvad.Vad(1)  # Aggressiveness level 1 (less aggressive)
            self.logger.info("WebRTC VAD initialized with aggressiveness level 1")

            # Setup input stream
            if self.config.input_device is not None or self._has_input_device():
                self._setup_input_stream()
            else:
                self.logger.warning("No input device available")

            # Setup output stream
            if self.config.output_device is not None or self._has_output_device():
                self._setup_output_stream()
            else:
                self.logger.warning("No output device available")

            self.logger.info("Audio manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize audio manager: {e}")
            raise

    def _has_input_device(self) -> bool:
        """Check if an input device is available."""
        try:
            device_count = self.pyaudio.get_device_count()
            for i in range(device_count):
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking input devices: {e}")
            return False

    def _has_output_device(self) -> bool:
        """Check if an output device is available."""
        try:
            device_count = self.pyaudio.get_device_count()
            for i in range(device_count):
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info["maxOutputChannels"] > 0:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking output devices: {e}")
            return False

    def _setup_input_stream(self) -> None:
        """Setup the audio input stream."""
        try:
            self.input_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._input_callback,
            )
            self.logger.info("Audio input stream initialized")
        except Exception as e:
            self.logger.error(f"Failed to setup input stream: {e}")
            self.input_stream = None

    def _setup_output_stream(self) -> None:
        """Setup the audio output stream."""
        try:
            self.output_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                output_device_index=self.config.output_device,
                frames_per_buffer=self.config.chunk_size,
            )
            self.logger.info("Audio output stream initialized")
        except Exception as e:
            self.logger.error(f"Failed to setup output stream: {e}")
            self.output_stream = None

    def _input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream."""
        if status:
            self.logger.warning(f"Audio input callback status: {status}")

        # Only record if we're not speaking (prevent feedback)
        if self.is_recording and not self.is_speaking:
            self.input_buffer.append(in_data)

        # Call external callback if set
        if self.audio_callback and self.is_recording and not self.is_speaking:
            try:
                self.audio_callback(in_data)
            except Exception as e:
                self.logger.error(f"Error in audio callback: {e}")

        return (None, pyaudio.paContinue)

    async def listen(self) -> Optional[np.ndarray]:
        """
        Listen for audio input and return when speech is detected.

        Returns:
            Audio data as numpy array, or None if no speech detected
        """
        if not self.input_stream:
            self.logger.warning("No input stream available")
            return None

        # Start recording
        self.is_recording = True
        self.input_buffer.clear()

        # Wait for voice activity
        await self._wait_for_speech()

        # Stop recording
        self.is_recording = False

        if not self.input_buffer:
            return None

        # Convert buffer to numpy array
        audio_data = b"".join(self.input_buffer)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        return audio_array

    async def _wait_for_speech(self) -> None:
        """Wait for speech activity using VAD."""
        speech_frames = 0
        silence_frames = 0
        min_speech_frames = 5  # Increased minimum frames to avoid false positives
        max_silence_frames = 50  # Increased silence tolerance

        # Wait for initial speech detection
        speech_detected = False

        while self.is_recording:
            # Skip processing if we're currently speaking (TTS output)
            if self.is_speaking:
                await asyncio.sleep(0.1)  # Longer delay when speaking
                continue

            # Implement cooldown period after TTS ends
            import time

            if time.time() - self.last_speech_end_time < 3.0:  # 3-second cooldown
                await asyncio.sleep(0.1)
                continue

            if len(self.input_buffer) > 0:
                # Get latest audio frame
                frame = self.input_buffer[-1]

                # Check voice activity
                is_speech = self._is_speech(frame)

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                    speech_detected = True
                    self.logger.debug(f"Speech detected (frames: {speech_frames})")
                else:
                    silence_frames += 1
                    if speech_detected:
                        self.logger.debug(
                            f"Silence detected (frames: {silence_frames})"
                        )

                # Stop if we have enough speech followed by silence
                if (
                    speech_detected
                    and speech_frames >= min_speech_frames
                    and silence_frames >= max_silence_frames
                ):
                    self.logger.info(
                        f"Speech complete: {speech_frames} speech frames, {silence_frames} silence frames"
                    )
                    break

                # Also stop if we've been recording for too long without speech
                if (
                    not speech_detected and len(self.input_buffer) > 480
                ):  # ~5 seconds at 16kHz (increased timeout)
                    self.logger.warning("No speech detected, stopping recording")
                    break

            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

    def _is_speech(self, frame: bytes) -> bool:
        """
        Check if audio frame contains speech using VAD.

        Args:
            frame: Audio frame bytes

        Returns:
            True if speech is detected
        """
        try:
            # VAD expects specific frame sizes (10, 20, or 30 ms)
            # Calculate frame size for 20ms at current sample rate
            frame_size_20ms = int(self.config.sample_rate * 0.02)

            if len(frame) >= frame_size_20ms * 2:  # 2 bytes per sample (16-bit)
                frame_20ms = frame[: frame_size_20ms * 2]
                return self.vad.is_speech(frame_20ms, self.config.sample_rate)

            return False
        except Exception as e:
            self.logger.debug(f"VAD error: {e}")
            return True  # Assume speech if VAD fails

    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = None) -> None:
        """
        Play audio data through the output stream.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data (for resampling if needed)
        """
        if not self.output_stream:
            self.logger.warning("No output stream available")
            return

        try:
            # Set speaking flag to prevent audio feedback
            self.is_speaking = True
            self.logger.debug(
                "Audio playback started - input disabled to prevent feedback"
            )

            # Resample if needed (simple approach for now)
            processed_audio = audio_data
            if sample_rate and sample_rate != self.config.sample_rate:
                self.logger.debug(
                    f"Resampling audio from {sample_rate}Hz to {self.config.sample_rate}Hz"
                )
                # Simple resampling by duplicating/dropping samples
                ratio = self.config.sample_rate / sample_rate
                if ratio != 1.0:
                    new_length = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, new_length).astype(
                        int
                    )
                    processed_audio = audio_data[indices]

            # Convert to bytes
            audio_bytes = processed_audio.astype(np.int16).tobytes()

            # Play audio in chunks
            chunk_size = self.config.chunk_size * 2  # 2 bytes per sample
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i: i + chunk_size]
                self.output_stream.write(chunk)

            # Calculate proper delay based on audio length
            duration = len(processed_audio) / self.config.sample_rate
            await asyncio.sleep(max(0.1, duration + 0.2))  # Audio duration + buffer
            self.logger.debug(f"Audio playback completed ({duration:.2f}s)")

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
        finally:
            # Clear speaking flag to re-enable input
            self.is_speaking = False
            # Clear any residual audio that might have been captured during playback
            self.clear_input_buffer()
            self.logger.debug("Audio playback finished - input re-enabled")

    def start_recording(self) -> None:
        """Start continuous recording."""
        if self.input_stream:
            self.input_stream.start_stream()
            self.is_recording = True
            self.logger.info("Started recording")

    def stop_recording(self) -> None:
        """Stop recording."""
        self.is_recording = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.logger.info("Stopped recording")

    async def cleanup(self) -> None:
        """Cleanup audio resources."""
        self.logger.info("Cleaning up audio manager...")

        self.is_recording = False
        self.is_speaking = False

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        if self.pyaudio:
            self.pyaudio.terminate()

        self.logger.info("Audio manager cleanup complete")

    def set_speaking_state(self, is_speaking: bool) -> None:
        """
        Set the speaking state to prevent audio feedback.

        Args:
            is_speaking: True if TTS is currently speaking, False otherwise
        """
        import time

        self.is_speaking = is_speaking
        if is_speaking:
            self.logger.debug("Speaking state enabled - audio input disabled")
            # Clear buffer immediately when starting to speak
            self.clear_input_buffer()
        else:
            # Record when speaking ended for cooldown period
            self.last_speech_end_time = time.time()
            self.logger.debug(
                "Speaking state disabled - audio input re-enabled with cooldown"
            )
            # Clear any residual audio in the buffer when re-enabling input
            self.clear_input_buffer()

    def clear_input_buffer(self) -> None:
        """Clear the audio input buffer to remove any residual audio."""
        self.input_buffer.clear()
        self.logger.debug("Audio input buffer cleared")

    def get_device_info(self) -> dict:
        """
        Get information about available audio devices.

        Returns:
            Dictionary containing device information
        """
        if not self.pyaudio:
            return {}

        devices = {
            "input_devices": [],
            "output_devices": [],
            "default_input": self.pyaudio.get_default_input_device_info(),
            "default_output": self.pyaudio.get_default_output_device_info(),
        }

        device_count = self.pyaudio.get_device_count()
        for i in range(device_count):
            device_info = self.pyaudio.get_device_info_by_index(i)

            if device_info["maxInputChannels"] > 0:
                devices["input_devices"].append(
                    {
                        "index": i,
                        "name": device_info["name"],
                        "channels": device_info["maxInputChannels"],
                    }
                )

            if device_info["maxOutputChannels"] > 0:
                devices["output_devices"].append(
                    {
                        "index": i,
                        "name": device_info["name"],
                        "channels": device_info["maxOutputChannels"],
                    }
                )

        return devices
