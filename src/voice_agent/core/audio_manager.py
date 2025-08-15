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

# ---------------------------------------------------------------------------
# Low-level ALSA / JACK noise suppression helpers
# ---------------------------------------------------------------------------
from contextlib import contextmanager
import os
import ctypes
from ctypes import CFUNCTYPE, c_char_p, c_int


@contextmanager
def _suppress_alsa_jack_errors(enabled: bool = True):
    """
    Suppress stderr noise from ALSA / JACK / PortAudio initialization.
    Strategy:
      1. Attempt to set ALSA error handler via snd_lib_error_set_handler to a no-op.
      2. Fallback to temporarily redirecting file descriptor 2 (stderr) to /dev/null.
    Only active when `enabled` is True (i.e., not in debug mode).
    """
    if not enabled:
        # Debug mode – do not suppress.
        yield
        return

    # Try native ALSA handler first
    reset_needed = False
    c_err_handler = None
    devnull = None
    old_stderr_fd = None
    try:
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

        def _py_alsa_error_handler(filename, line, function, err, fmt):
            # Intentionally ignore all ALSA messages
            return

        try:
            asound = ctypes.cdll.LoadLibrary("libasound.so")
            c_err_handler = ERROR_HANDLER_FUNC(_py_alsa_error_handler)
            asound.snd_lib_error_set_handler(c_err_handler)
            reset_needed = True
            yield
        except Exception:
            # Fallback: redirect stderr
            try:
                devnull = open(os.devnull, "w")
                old_stderr_fd = os.dup(2)
                os.dup2(devnull.fileno(), 2)
            except Exception:
                # If even this fails, just proceed without suppression
                pass
            yield
        finally:
            # Restore ALSA handler
            if reset_needed:
                try:
                    asound.snd_lib_error_set_handler(None)  # type: ignore
                except Exception:
                    pass
            # Restore stderr
            if old_stderr_fd is not None:
                try:
                    os.dup2(old_stderr_fd, 2)
                except Exception:
                    pass
            if devnull is not None:
                try:
                    devnull.close()
                except Exception:
                    pass
    except Exception:
        # Absolute fallback: do nothing
        yield


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

        # PyAudio components
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

        # Voice activity detection
        self.vad: Optional[webrtcvad.Vad] = None
        # Most recent normalized input level (0.0–1.0) updated in _input_callback for UI meter
        self.last_level: float = 0.0

        # Audio buffers
        self.input_buffer: List[bytes] = []
        self.is_recording = False
        self.is_speaking = False  # Flag to prevent feedback during TTS
        self.last_speech_end_time = 0.0  # Track when TTS last ended

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
        self.logger.info("Initializing audio manager...")
        self._emit_state("audio_input", "initializing", "Initializing audio")
        self._emit_state("audio_output", "initializing", "Initializing audio")

        try:
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            self.logger.info("PyAudio initialized")

            # Initialize VAD with configured aggressiveness
            vad_level = getattr(self.config, "vad_aggressiveness", 1)
            try:
                self.vad = webrtcvad.Vad(int(vad_level))
            except Exception:
                self.logger.warning(
                    f"Invalid vad_aggressiveness={vad_level}, falling back to 1"
                )
                self.vad = webrtcvad.Vad(1)
            self.logger.info(
                f"WebRTC VAD initialized with aggressiveness level {vad_level}"
            )

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

            self._emit_state(
                "audio_input",
                "ready" if self.input_stream else "disabled",
                None if self.input_stream else "No input device",
            )
            self._emit_state(
                "audio_output",
                "ready" if self.output_stream else "disabled",
                None if self.output_stream else "No output device",
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize audio manager: {e}")
            self._emit_state("audio_input", "error", str(e))
            self._emit_state("audio_output", "error", str(e))
            raise

    def _has_input_device(self) -> bool:
        """Check if an input device is available."""
        if self.pyaudio is None:
            return False
        try:
            device_count = self.pyaudio.get_device_count()
            for i in range(device_count):
                device_info = self.pyaudio.get_device_info_by_index(i)
                try:
                    max_in = int(float(device_info.get("maxInputChannels", 0) or 0))
                except (ValueError, TypeError):
                    max_in = 0
                if max_in > 0:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking input devices: {e}")
            return False

    def _has_output_device(self) -> bool:
        """Check if an output device is available."""
        if self.pyaudio is None:
            return False
        try:
            device_count = self.pyaudio.get_device_count()
            for i in range(device_count):
                device_info = self.pyaudio.get_device_info_by_index(i)
                try:
                    max_out = int(float(device_info.get("maxOutputChannels", 0) or 0))
                except (ValueError, TypeError):
                    max_out = 0
                if max_out > 0:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking output devices: {e}")
            return False

    def _setup_input_stream(self) -> None:
        """Setup the audio input stream."""
        if self.pyaudio is None:
            self.logger.error("PyAudio not initialized; cannot open input stream")
            return
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
        if self.pyaudio is None:
            self.logger.error("PyAudio not initialized; cannot open output stream")
            return
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

        # Update level meter (RMS scaled) irrespective of recording state
        try:
            samples = np.frombuffer(in_data, dtype=np.int16)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                # Scale: speech RMS typically up to ~10000; clamp to 1.0
                self.last_level = max(0.0, min(rms / 12000.0, 1.0))
        except Exception:
            # Non-fatal; keep previous value
            pass

        # Only record if we're not speaking (prevent feedback)
        if self.is_recording and not self.is_speaking:
            self.input_buffer.append(in_data)

            # Call external callback if set
            if callable(self.audio_callback):
                try:
                    self.audio_callback(in_data)  # type: ignore[call-arg]
                except Exception as e:
                    self.logger.error(f"Error in audio callback: {e}")
        else:
            # While speaking, monitor for possible barge-in (user attempting to interrupt)
            if self.is_speaking and self._barge_in_callback:
                try:
                    # Heuristic: sustained elevated input level while TTS is active
                    # Use previously computed self.last_level (> ~0.28) for N consecutive callbacks
                    if self.last_level > 0.28:
                        self._interrupt_energy_frames += 1
                    else:
                        self._interrupt_energy_frames = 0
                    if (
                        not self._barge_in_triggered
                        and self._interrupt_energy_frames >= 5
                    ):
                        self._barge_in_triggered = True
                        self.logger.info(
                            "Barge-in detected (user speech during TTS) – requesting interruption"
                        )
                        try:
                            self._barge_in_callback()
                        except Exception:
                            self.logger.debug(
                                "Error invoking barge-in callback", exc_info=True
                            )
                except Exception:
                    self.logger.debug("Barge-in detection error", exc_info=True)

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
            import time

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

                # Timeout: no confirmed speech after buffer grows large
                if (
                    not speech_started and len(self.input_buffer) > 480
                ):  # ~5 seconds at 16kHz (increased timeout)
                    self.logger.debug(
                        "No confirmed speech detected within timeout window"
                    )
                    break

            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

    def _is_speech(self, frame: bytes) -> bool:
        """
        Check if audio frame contains speech using VAD plus simple energy gate.

        Args:
            frame: Audio frame bytes

        Returns:
            True if speech is detected
        """
        try:
            # If VAD not initialized, conservatively return False (avoid false positives)
            if self.vad is None:
                return False

            # VAD expects specific frame sizes (10, 20, or 30 ms)
            frame_size_20ms = int(self.config.sample_rate * 0.02)
            if (
                len(frame) < frame_size_20ms * 2
            ):  # Need enough samples (2 bytes per int16)
                return False

            frame_20ms = frame[: frame_size_20ms * 2]

            # Simple amplitude (energy) gate to filter out very quiet clicks
            # Convert to int16 numpy array
            samples = np.frombuffer(frame_20ms, dtype=np.int16)
            # Root mean square amplitude
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            # Threshold (empirical): ignore frames with rms below ~250 (very quiet)
            if rms < 250:
                return False

            return self.vad.is_speech(frame_20ms, self.config.sample_rate)
        except Exception as e:
            self.logger.debug(f"VAD error: {e}")
            # Previous behavior returned True (treat errors as speech); for noise suppression we choose False
            return False

    async def play_audio(
        self, audio_data: np.ndarray, sample_rate: Optional[int] = None
    ) -> None:
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

            # Convert to bytes
            audio_bytes = processed_audio.astype(np.int16).tobytes()

            # Play audio in chunks
            chunk_size = self.config.chunk_size * 2  # 2 bytes per sample
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                self.output_stream.write(chunk)
                # Yield to event loop so input callbacks & other tasks run
                await asyncio.sleep(0)
                # Early interruption check (barge-in / TTS stop)
                if (
                    self._should_interrupt_playback
                    and self._should_interrupt_playback()
                ):
                    self.logger.info("Playback interrupted early by barge-in request")
                    break

            # If not interrupted, allow buffer drain (short sleep proportional to remaining audio length)
            if not (
                self._should_interrupt_playback and self._should_interrupt_playback()
            ):
                duration = len(processed_audio) / self.config.sample_rate
                await asyncio.sleep(max(0.05, min(0.25, duration * 0.05)))
                self.logger.debug(f"Audio playback completed ({duration:.2f}s)")

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
        finally:
            # Clear speaking flag to re-enable input
            self.is_speaking = False
            # Record end time for cooldown logic and re-enable input
            import time

            self.last_speech_end_time = time.time()
            # Clear any residual audio that might have been captured during playback
            self.clear_input_buffer()
            # Reset barge-in detection counters
            self._barge_in_triggered = False
            self._interrupt_energy_frames = 0
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

        # Safely obtain default device info (PyAudio may raise if unavailable)
        try:
            default_input = self.pyaudio.get_default_input_device_info()
        except Exception:
            default_input = None
        try:
            default_output = self.pyaudio.get_default_output_device_info()
        except Exception:
            default_output = None

        devices = {
            "input_devices": [],
            "output_devices": [],
            "default_input": default_input,
            "default_output": default_output,
        }

        device_count = self.pyaudio.get_device_count()
        for i in range(device_count):
            device_info = self.pyaudio.get_device_info_by_index(i)

            # Some backends may return numeric fields as float or string; normalize
            try:
                max_input = int(float(device_info.get("maxInputChannels", 0) or 0))
            except (ValueError, TypeError):
                max_input = 0
            try:
                max_output = int(float(device_info.get("maxOutputChannels", 0) or 0))
            except (ValueError, TypeError):
                max_output = 0

            if max_input > 0:
                devices["input_devices"].append(
                    {
                        "index": i,
                        "name": device_info.get("name", f"Input {i}"),
                        "channels": max_input,
                    }
                )

            if max_output > 0:
                devices["output_devices"].append(
                    {
                        "index": i,
                        "name": device_info.get("name", f"Output {i}"),
                        "channels": max_output,
                    }
                )

        return devices
