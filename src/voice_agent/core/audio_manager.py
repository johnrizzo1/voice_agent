"""
Audio input/output management for the voice agent.
"""

import asyncio
import logging
import time
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
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

    def __init__(
        self,
        config: AudioConfig,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the audio manager.

        Args:
            config: Audio configuration settings
            state_callback: Optional callback(component, state, message) to report status
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # SoundDevice components
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None

        # Voice activity detection
        self.vad: Optional[webrtcvad.Vad] = None
        # Most recent normalized input level (0.0–1.0) updated in _input_callback for UI meter
        self.last_level: float = 0.0

        # Audio buffers
        self.input_buffer: List[bytes] = []
        self.is_recording = False
        self.is_speaking = False  # Flag to prevent feedback during TTS
        self.last_speech_end_time = 0.0  # Track when TTS last ended

        # Microphone state management
        self.is_microphone_muted = False  # User-controlled mute state
        self.is_microphone_paused = False  # Temporary pause state
        self.microphone_error = False  # Error state

        # Callbacks
        self.audio_callback: Optional[Callable] = None
        self._state_callback = state_callback

        # Barge-in / interruption support
        self._barge_in_callback: Optional[Callable[[], None]] = None
        self._should_interrupt_playback: Optional[Callable[[], bool]] = None
        self._barge_in_triggered: bool = False
        self._interrupt_energy_frames: int = (
            0  # successive high-energy frames while speaking
        )

    def set_state_callback(
        self, cb: Optional[Callable[[str, str, Optional[str]], None]]
    ) -> None:
        """Set or replace the pipeline state callback."""
        self._state_callback = cb

    # ---------------- Barge-in / interruption wiring ----------------
    def set_barge_in_callback(self, cb: Optional[Callable[[], None]]) -> None:
        """
        Register a callback invoked when user speech is detected during TTS playback.
        The callback should request interruption on the active TTS service.
        """
        self._barge_in_callback = cb

    def set_interrupt_getter(self, cb: Optional[Callable[[], bool]]) -> None:
        """
        Register a predicate polled during audio playback to decide early stop.

        Args:
            cb: Callable returning True if current playback should be interrupted.
        """
        self._should_interrupt_playback = cb

    def _emit_state(
        self, component: str, state: str, message: Optional[str] = None
    ) -> None:
        """Emit a pipeline component state update if callback is set."""
        if self._state_callback:
            try:
                self._state_callback(component, state, message)
            except Exception:
                # Never let UI callback errors break audio
                self.logger.debug("State callback error", exc_info=True)

    async def initialize(self) -> None:
        """Initialize audio components."""
        self.logger.info("Initializing audio manager with SoundDevice...")
        self._emit_state("audio_input", "initializing", "Initializing audio")
        self._emit_state("audio_output", "initializing", "Initializing audio")

        try:
            # Initialize VAD with configured aggressiveness
            vad_level = getattr(self.config, "vad_aggressiveness", 1)
            try:
                self.vad = webrtcvad.Vad(int(vad_level))
            except Exception:
                self.logger.warning(
                    f"Invalid vad_aggressiveness={vad_level}, falling back to 1"
                )
                self.vad = webrtcvad.Vad(1)

            self.logger.info(f"VAD initialized with aggressiveness level {vad_level}")

            # Configure SoundDevice defaults
            sd.default.samplerate = self.config.sample_rate
            sd.default.channels = getattr(self.config, "channels", 1)
            sd.default.dtype = getattr(self.config, "dtype", "int16")

            # Set device defaults if specified
            if (
                hasattr(self.config, "input_device")
                and self.config.input_device is not None
            ):
                sd.default.device[0] = self.config.input_device
            if (
                hasattr(self.config, "output_device")
                and self.config.output_device is not None
            ):
                sd.default.device[1] = self.config.output_device

            # Test audio devices
            if not self._has_input_device():
                self.logger.error("No input device available")
                self._emit_state("audio_input", "error", "No microphone found")
                self.microphone_error = True
            else:
                self._emit_state("audio_input", "ready", "Microphone ready")

            if not self._has_output_device():
                self.logger.warning("No output device available")
                self._emit_state("audio_output", "error", "No speakers found")
            else:
                self._emit_state("audio_output", "ready", "Speakers ready")

            # Start passive input stream immediately so level meter updates even before a listen() call
            try:
                await self.start_input_stream()
            except Exception as _lvl_err:
                self.logger.debug(
                    f"Passive input stream start failed (meter will stay at 0 until listen): {_lvl_err}"
                )

            self.logger.info("SoundDevice audio manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize audio manager: {e}")
            self._emit_state(
                "audio_input", "error", f"Audio initialization failed: {e}"
            )
            self._emit_state(
                "audio_output", "error", f"Audio initialization failed: {e}"
            )
            raise

    def _has_input_device(self) -> bool:
        """Check if an input device is available."""
        try:
            devices = sd.query_devices()
            self.logger.debug(f"Enumerated audio devices: {devices}")
            for device in devices:
                if device["max_input_channels"] > 0:
                    self.logger.debug(f"Detected input device: {device}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking input devices: {e}")
            return False

    def _has_output_device(self) -> bool:
        """Check if an output device is available."""
        try:
            devices = sd.query_devices()
            for device in devices:
                if device["max_output_channels"] > 0:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking output devices: {e}")
            return False

    def _handle_audio_error(self, error: Exception) -> bool:
        """
        Unified error handling for all platforms.

        Args:
            error: The audio error that occurred

        Returns:
            True if error was handled and recovery attempted, False otherwise
        """
        if isinstance(error, sd.PortAudioError):
            self.logger.warning(f"PortAudio error: {error}")
            # Attempt fallback device selection
            return self._attempt_fallback_device()
        else:
            self.logger.error(f"Unexpected audio error: {error}")
            return False

    def _attempt_fallback_device(self) -> bool:
        """Attempt to fallback to default devices on error."""
        try:
            # Reset to system defaults
            sd.default.device = None
            self.logger.info("Reset to default audio devices")
            return True
        except Exception as e:
            self.logger.error(f"Failed to fallback to default devices: {e}")
            return False

    def _input_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """
        SoundDevice input callback for processing audio data.

        Args:
            indata: Input audio data as numpy array
            frames: Number of frames
            time_info: Time information
            status: Stream status
        """
        try:
            if status:
                self.logger.debug(f"Input callback status: {status}")

            # Skip processing if microphone is muted or paused
            if self.is_microphone_muted or self.is_microphone_paused:
                return

            # Convert numpy array to bytes for compatibility with existing code
            audio_bytes = indata.astype(np.int16).tobytes()

            # Update audio level for UI meter
            if len(indata) > 0:
                # Normalize samples to [-1, 1], compute RMS, then apply light smoothing
                samples = indata.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples**2)) + 1e-8)
                smoothed = (0.7 * self.last_level) + (0.3 * rms)
                self.last_level = max(0.0, min(1.0, smoothed))
                if self.last_level < 0.005:  # noise floor clamp
                    self.last_level = 0.0
                self._emit_state(
                    "voice_meter", "active", f"Audio level={self.last_level:.2f}"
                )
                # Throttle detailed debug logging (approx every 0.1s)
                if int(time.time() * 10) % 10 == 0:
                    self.logger.debug(
                        f"Audio level update: raw_rms={rms:.4f} level={self.last_level:.3f}"
                    )

            # Add to buffer if recording
            if self.is_recording:
                self.input_buffer.append(audio_bytes)

            # Barge-in detection during TTS playback
            if self.is_speaking and getattr(self.config, "barge_in_enabled", True):
                try:
                    # Calculate audio energy
                    energy = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
                    barge_in_threshold = getattr(
                        self.config, "barge_in_energy_threshold", 0.28
                    )

                    if energy > barge_in_threshold:
                        self._interrupt_energy_frames += 1
                        consecutive_frames = getattr(
                            self.config, "barge_in_consecutive_frames", 5
                        )

                        if (
                            self._interrupt_energy_frames >= consecutive_frames
                            and not self._barge_in_triggered
                        ):
                            self._barge_in_triggered = True
                            self.logger.info(
                                f"Barge-in detected: energy={energy:.3f}, frames={self._interrupt_energy_frames}"
                            )

                            if self._barge_in_callback:
                                try:
                                    self._barge_in_callback()
                                except Exception:
                                    self.logger.debug(
                                        "Error invoking barge-in callback",
                                        exc_info=True,
                                    )
                    else:
                        self._interrupt_energy_frames = 0
                except Exception:
                    self.logger.debug("Barge-in detection error", exc_info=True)

        except Exception as e:
            self.logger.debug(f"Input callback error: {e}")

    async def start_input_stream(self) -> None:
        """Start the audio input stream."""
        if self.input_stream is not None:
            return

        try:
            # Calculate appropriate blocksize for VAD compatibility
            # Use chunk_size directly for VAD compatibility (320 samples = 10ms at 16kHz)
            frames_per_buffer = self.config.chunk_size

            # Ensure blocksize matches VAD requirements for optimal speech detection
            # VAD expects specific frame sizes: 160 (10ms), 320 (20ms), or 480 (30ms) samples
            vad_compatible_sizes = [160, 320, 480]
            if frames_per_buffer in vad_compatible_sizes:
                blocksize = frames_per_buffer
                self.logger.info(f"Using VAD-compatible blocksize: {blocksize} samples")
            else:
                # Fall back to closest VAD-compatible size
                blocksize = min(
                    vad_compatible_sizes, key=lambda x: abs(x - frames_per_buffer)
                )
                self.logger.warning(
                    f"Adjusted blocksize from {frames_per_buffer} to {blocksize} for VAD compatibility"
                )

            self.logger.debug(f"Starting input stream with blocksize: {blocksize}")

            # More robust stream creation with better error handling
            stream_params = {
                "samplerate": self.config.sample_rate,
                "channels": getattr(self.config, "channels", 1),
                "dtype": getattr(self.config, "dtype", "int16"),
                "blocksize": blocksize,
                "device": getattr(self.config, "input_device", None),
                "callback": self._input_callback,
                "latency": getattr(self.config, "latency", "low"),
            }
            self.logger.debug(
                f"Stream parameters set with latency={self.config.latency}"
            )

            # Only add CoreAudioSettings if available and we're on macOS
            try:
                import platform

                if platform.system() == "Darwin" and hasattr(sd, "CoreAudioSettings"):
                    stream_params["extra_settings"] = sd.CoreAudioSettings()
            except Exception:
                pass  # Ignore if platform detection fails

            self.input_stream = sd.InputStream(**stream_params)
            self.input_stream.start()
            self.logger.info(
                f"Input stream started successfully with blocksize {blocksize}"
            )
            self._emit_state("audio_input", "active", "Recording active")

        except Exception as e:
            self.logger.error(f"Failed to start input stream: {e}")
            # Try multiple fallback strategies
            fallback_configs = [
                # Fallback 1: Minimal parameters
                {
                    "samplerate": self.config.sample_rate,
                    "channels": 1,
                    "dtype": "int16",
                    "callback": self._input_callback,
                },
                # Fallback 2: Different blocksize
                {
                    "samplerate": self.config.sample_rate,
                    "channels": 1,
                    "dtype": "int16",
                    "blocksize": 1024,
                    "callback": self._input_callback,
                },
                # Fallback 3: Default device explicitly
                {
                    "samplerate": self.config.sample_rate,
                    "channels": 1,
                    "dtype": "int16",
                    "device": None,  # Use default
                    "callback": self._input_callback,
                },
            ]

            for i, fallback_config in enumerate(fallback_configs, 1):
                try:
                    self.logger.info(
                        f"Attempting fallback input stream initialization {i}/3..."
                    )
                    self.input_stream = sd.InputStream(**fallback_config)
                    self.input_stream.start()
                    self.logger.info(f"Fallback input stream {i} started successfully")
                    self._emit_state(
                        "audio_input", "active", f"Recording active (fallback mode {i})"
                    )
                    return  # Success!
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback {i} failed: {fallback_error}")
                    continue

            # All fallbacks failed
            self.logger.error("All input stream initialization attempts failed")
            self._handle_audio_error(e)
            raise e

    async def stop_input_stream(self) -> None:
        """Stop the audio input stream."""
        if self.input_stream is not None:
            try:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
                self.logger.info("Input stream stopped")
                self._emit_state("audio_input", "ready", "Microphone ready")
            except Exception as e:
                self.logger.error(f"Error stopping input stream: {e}")

    async def listen(self) -> Optional[np.ndarray]:
        """
        Listen for audio input and return when speech is detected.

        Returns:
            Audio data as numpy array, or None if no speech detected
        """
        try:
            if not self.input_stream:
                await self.start_input_stream()

            if not self.input_stream:
                self.logger.warning("No input stream available after start attempt")
                return None

            # Start recording
            self.is_recording = True
            self.input_buffer.clear()
            self._emit_state("audio_input", "active", "listening")

            # Wait for voice activity
            await self._wait_for_speech()

            # Stop recording
            self.is_recording = False

            if not self.input_buffer:
                self._emit_state("audio_input", "ready", None)
                return None

            # Convert buffer to numpy array
            audio_data = b"".join(self.input_buffer)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            self._emit_state("audio_input", "ready", None)
            return audio_array

        except Exception as e:
            self.logger.error(f"Error during listen operation: {e}")
            self.is_recording = False
            self._emit_state("audio_input", "error", f"Listen failed: {e}")
            # Don't re-raise - return None to allow graceful fallback
            return None

    async def _wait_for_speech(self) -> None:
        """Wait for speech activity using VAD with stricter start criteria to avoid false triggers (e.g. keyboard clicks)."""
        speech_frames = 0  # Count of speech-classified frames (not yet confirmed start)
        silence_frames = 0  # Count of silence frames after speech start
        speech_started = False  # Becomes True only after min_speech_frames consecutive (or near-consecutive) speech frames

        # Pull configurable thresholds with safe defaults
        min_speech_frames = getattr(self.config, "min_speech_frames", 5)
        max_silence_frames = getattr(self.config, "max_silence_frames", 50)
        cooldown = getattr(self.config, "speech_detection_cooldown", 1.0)

        while self.is_recording:
            # Skip processing if we're currently speaking (TTS output)
            if self.is_speaking:
                await asyncio.sleep(0.1)  # Longer delay when speaking
                continue

            # Implement cooldown period after TTS ends
            if time.time() - self.last_speech_end_time < cooldown:
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
                    # Only declare speech started after enough consecutive frames
                    if not speech_started and speech_frames >= min_speech_frames:
                        speech_started = True
                        self.logger.debug(
                            f"Speech start confirmed after {speech_frames} frames"
                        )
                    elif speech_started:
                        # Ongoing speech (optional detailed logging can be throttled)
                        self.logger.debug(
                            f"Ongoing speech (speech_frames={speech_frames})"
                        )
                else:
                    if speech_started:
                        silence_frames += 1
                        # Log only occasional silence frames to reduce spam
                        if silence_frames in (1, max_silence_frames // 2):
                            self.logger.debug(
                                f"Silence after speech (silence_frames={silence_frames})"
                            )
                    else:
                        # Reset tentative speech counter if we haven't confirmed start
                        speech_frames = 0

                # Stop when confirmed speech has ended with required trailing silence
                if speech_started and silence_frames >= max_silence_frames:
                    self.logger.info(
                        f"Speech complete: {speech_frames} speech frames, {silence_frames} trailing silence frames"
                    )
                    break

            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

    def _is_speech(self, frame: bytes) -> bool:
        """
        Determine if the given audio frame contains speech using VAD and energy threshold.

        Args:
            frame: Audio frame as bytes

        Returns:
            True if frame likely contains speech, False otherwise
        """
        try:
            # Energy-based filtering first
            if len(frame) < 2:
                return False

            audio_array = np.frombuffer(frame, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            energy_threshold = getattr(self.config, "energy_threshold", 7000)

            if rms < energy_threshold:
                return False

            # VAD check (requires specific frame sizes: 10ms, 20ms, or 30ms at 16kHz)
            if self.vad and len(frame) in [320, 640, 960]:  # Valid frame sizes for VAD
                return self.vad.is_speech(frame, self.config.sample_rate)
            else:
                # Fallback to energy-based detection
                return True

        except Exception as e:
            self.logger.debug(f"VAD error: {e}")
            # Previous behavior returned True (treat errors as speech); for noise suppression we choose False
            return False

    async def play_audio(
        self, audio_data: np.ndarray, sample_rate: Optional[int] = None
    ) -> None:
        """
        Play audio data through SoundDevice.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data (for resampling if needed)
        """
        try:
            # Set speaking flag to prevent audio feedback (if enabled)
            feedback_prevention = getattr(
                self.config, "feedback_prevention_enabled", True
            )
            self.is_speaking = feedback_prevention
            if feedback_prevention:
                self.logger.debug(
                    "Audio playback started - input disabled to prevent feedback"
                )

            # Clear input buffer if configured
            if getattr(self.config, "buffer_clear_on_playback", True):
                self.clear_input_buffer()

            # Resample if needed (simple approach for now)
            processed_audio = audio_data
            if sample_rate and sample_rate != self.config.sample_rate:
                self.logger.debug(
                    "Resampling audio from %dHz to %dHz",
                    sample_rate,
                    self.config.sample_rate,
                )
                # Simple resampling by duplicating/dropping samples
                ratio = self.config.sample_rate / sample_rate
                if ratio != 1.0:
                    new_length = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, new_length).astype(
                        int
                    )
                    processed_audio = audio_data[indices]

            # Ensure proper data type
            if processed_audio.dtype != np.int16:
                processed_audio = processed_audio.astype(np.int16)

            # Play audio using SoundDevice with smooth streaming
            duration = len(processed_audio) / self.config.sample_rate
            self.logger.debug(
                f"Starting audio playback ({duration:.2f}s, {len(processed_audio)} samples)"
            )

            # Use SoundDevice's streaming playback for smooth audio
            sd.play(
                processed_audio,
                samplerate=self.config.sample_rate,
                device=getattr(self.config, "output_device", None),
                blocking=False,
            )

            # Wait for playback to complete with interruption support
            start_time = time.time()
            while sd.get_stream().active:
                await asyncio.sleep(0.01)

                # Check for early interruption (barge-in / TTS stop)
                if (
                    self._should_interrupt_playback
                    and self._should_interrupt_playback()
                ):
                    self.logger.info("Playback interrupted early by barge-in request")
                    sd.stop()
                    break

                # Safety timeout to prevent infinite loops
                if time.time() - start_time > duration + 2.0:
                    self.logger.warning("Audio playback timeout - stopping")
                    sd.stop()
                    break

            # Final wait for completion if not interrupted
            if not (
                self._should_interrupt_playback and self._should_interrupt_playback()
            ):
                sd.wait()
                self.logger.debug(f"Audio playback completed ({duration:.2f}s)")

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
        finally:
            # Clear speaking flag to re-enable input
            self.is_speaking = False
            # Record end time for cooldown logic and re-enable input
            self.last_speech_end_time = time.time()
            # Clear any residual audio that might have been captured during playback
            self.clear_input_buffer()

            # Optional double buffer clear for stubborn feedback issues
            if getattr(self.config, "double_buffer_clear", False):
                await asyncio.sleep(0.1)  # Brief delay
                self.clear_input_buffer()
                self.logger.debug("Double buffer clear performed")

            # Reset barge-in detection counters
            self._barge_in_triggered = False
            self._interrupt_energy_frames = 0
            self.logger.debug("Audio playback finished - input re-enabled")

    def start_recording(self) -> None:
        """Start recording audio (legacy compatibility method)."""
        asyncio.create_task(self.start_input_stream())

    def stop_recording(self) -> None:
        """Stop recording audio (legacy compatibility method)."""
        asyncio.create_task(self.stop_input_stream())

    def clear_input_buffer(self) -> None:
        """Clear the input audio buffer."""
        self.input_buffer.clear()
        self.logger.debug("Input buffer cleared")

    def toggle_microphone_mute(self) -> bool:
        """
        Toggle microphone mute state.

        Returns:
            New mute state (True = muted, False = unmuted)
        """
        self.is_microphone_muted = not self.is_microphone_muted
        state = "muted" if self.is_microphone_muted else "unmuted"
        self.logger.info(f"Microphone {state}")
        self._emit_state(
            "audio_input",
            "muted" if self.is_microphone_muted else "ready",
            f"Microphone {state}",
        )
        return self.is_microphone_muted

    def pause_microphone(self) -> None:
        """Temporarily pause microphone input."""
        self.is_microphone_paused = True
        self.logger.debug("Microphone paused")
        self._emit_state("audio_input", "paused", "Microphone paused")

    def resume_microphone(self) -> None:
        """Resume microphone input after pause."""
        self.is_microphone_paused = False
        self.logger.debug("Microphone resumed")
        self._emit_state("audio_input", "ready", "Microphone resumed")

    async def cleanup(self) -> None:
        """Clean up audio resources."""
        self.logger.info("Cleaning up audio manager...")

        # Stop streams
        await self.stop_input_stream()

        # Stop any active playback
        try:
            sd.stop()
        except Exception:
            pass

        self.logger.info("Audio manager cleanup complete")

    def get_status(self) -> dict:
        """
        Get current audio manager status.

        Returns:
            Dictionary containing status information
        """
        return {
            "input_stream_active": self.input_stream is not None
            and self.input_stream.active,
            "is_recording": self.is_recording,
            "is_speaking": self.is_speaking,
            "is_muted": self.is_microphone_muted,
            "is_paused": self.is_microphone_paused,
            "microphone_error": self.microphone_error,
            "input_available": self.input_stream is not None,
            "last_level": self.last_level,
        }

    def get_device_info(self) -> dict:
        """
        Get information about available audio devices.

        Returns:
            Dictionary containing device information
        """
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]

            device_info = {
                "input_devices": [],
                "output_devices": [],
                "default_input": default_input,
                "default_output": default_output,
            }

            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    device_info["input_devices"].append(
                        {
                            "index": i,
                            "name": device["name"],
                            "channels": device["max_input_channels"],
                        }
                    )

                if device["max_output_channels"] > 0:
                    device_info["output_devices"].append(
                        {
                            "index": i,
                            "name": device["name"],
                            "channels": device["max_output_channels"],
                        }
                    )

            return device_info

        except Exception as e:
            self.logger.error(f"Error getting device info: {e}")
            return {}

    def set_speaking_state(self, speaking: bool) -> None:
        """
        Set the speaking state (compatibility method for TTS service).

        Args:
            speaking: True if TTS is currently speaking, False otherwise
        """
        self.is_speaking = speaking
        if speaking:
            self.logger.debug(
                "Speaking state set - input disabled for feedback prevention"
            )
        else:
            self.logger.debug("Speaking state cleared - input re-enabled")
