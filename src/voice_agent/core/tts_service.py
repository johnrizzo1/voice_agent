"""
Text-to-Speech service for the voice agent.
"""

import asyncio
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from .config import TTSConfig

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import torch
    from TTS.api import TTS

    COQUI_AVAILABLE = True
except ImportError:
    TTS = None
    torch = None
    COQUI_AVAILABLE = False

try:
    import shutil
    import subprocess

    # Check if espeak-ng is installed
    ESPEAK_AVAILABLE = shutil.which("espeak-ng") is not None
except ImportError:
    ESPEAK_AVAILABLE = False

try:
    import scipy.io.wavfile as wavfile
    import torch
    from bark import SAMPLE_RATE, generate_audio, preload_models

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False


class TTSService:
    """
    Text-to-Speech service with multiple backend support.

    Supports:
    - Bark (Suno's high-quality neural TTS)
    - Coqui TTS (neural voice synthesis)
    - eSpeak-NG (improved synthetic TTS)
    - pyttsx3 (cross-platform system TTS)
    """

    def __init__(
        self,
        config: TTSConfig,
        audio_manager=None,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the TTS service.

        Args:
            config: TTS configuration settings
            audio_manager: Optional audio manager for audio playback
            state_callback: Optional callback(component, state, message)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audio_manager = audio_manager
        self._state_callback = state_callback

        # Engine instances
        self.pyttsx3_engine: Optional[pyttsx3.Engine] = None
        self.coqui_tts: Optional[TTS] = None
        self.bark_models_loaded = False

        # Service state
        self.is_initialized = False
        self.current_backend = self._determine_backend()

        # Audio settings
        self.sample_rate = 22050  # Default sample rate

        # Interruption (barge-in) state
        self._interrupt_requested: bool = False

    def _determine_backend(self) -> str:
        """Determine which TTS backend to use based on availability and config."""
        # Prefer high-quality offline TTS engines in order of quality
        if BARK_AVAILABLE and self.config.engine in ["bark", "auto"]:
            return "bark"
        elif COQUI_AVAILABLE and self.config.engine in ["coqui", "auto"]:
            return "coqui"
        elif ESPEAK_AVAILABLE and self.config.engine in ["espeak", "auto"]:
            return "espeak"
        elif pyttsx3:
            return "pyttsx3"
        elif BARK_AVAILABLE:
            return "bark"
        elif COQUI_AVAILABLE:
            return "coqui"
        elif ESPEAK_AVAILABLE:
            return "espeak"
        else:
            self.logger.error("No TTS backend available")
            return "none"

    def set_state_callback(
        self, cb: Optional[Callable[[str, str, Optional[str]], None]]
    ) -> None:
        self._state_callback = cb

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        if self._state_callback:
            try:
                self._state_callback("tts", state, message)
            except Exception:
                self.logger.debug("TTS state callback error", exc_info=True)

    # ---------------- Interruption (barge-in) API ----------------
    def request_interrupt(self) -> None:
        """
        Request that any ongoing TTS playback be interrupted.

        For pyttsx3 we attempt an engine.stop(); for streamed chunk playback
        (Bark / Coqui / eSpeak via AudioManager) a polling hook in AudioManager
        will terminate playback loop early.
        """
        if not self._interrupt_requested:
            self.logger.info("TTS interruption requested (barge-in)")
        self._interrupt_requested = True
        if self.current_backend == "pyttsx3" and self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except Exception:
                self.logger.debug(
                    "pyttsx3 stop() failed during interruption", exc_info=True
                )

    async def initialize(self) -> None:
        """Initialize the TTS service and load engines."""
        self.logger.info(
            f"Initializing TTS service with backend: {self.current_backend}"
        )
        self._emit_state("initializing", f"backend={self.current_backend}")

        if self.current_backend == "bark":
            await self._initialize_bark()
        elif self.current_backend == "pyttsx3":
            await self._initialize_pyttsx3()
        elif self.current_backend == "coqui":
            await self._initialize_coqui()
        elif self.current_backend == "espeak":
            await self._initialize_espeak()
        else:
            self.logger.error("No TTS backend could be initialized")
            return

        self.is_initialized = True
        self.logger.info("TTS service initialized")
        self._emit_state("ready", None)

    async def _initialize_pyttsx3(self) -> None:
        """Initialize pyttsx3 engine."""
        try:
            self.logger.info("Initializing pyttsx3 TTS engine...")

            # Initialize in a separate thread
            loop = asyncio.get_event_loop()
            self.pyttsx3_engine = await loop.run_in_executor(None, pyttsx3.init)

            # Configure engine
            if self.pyttsx3_engine:
                # Set speech rate
                rate = self.pyttsx3_engine.getProperty("rate")
                new_rate = int(rate * self.config.speed)
                self.pyttsx3_engine.setProperty("rate", new_rate)
                self.logger.debug(f"Set speech rate to {new_rate} (original: {rate})")

                # Set voice if specified
                if self.config.voice != "default":
                    voices = self.pyttsx3_engine.getProperty("voices")
                    if voices:
                        for voice in voices:
                            if self.config.voice.lower() in voice.name.lower():
                                self.pyttsx3_engine.setProperty("voice", voice.id)
                                self.logger.info(f"Set voice to: {voice.name}")
                                break
                        else:
                            self.logger.warning(
                                f"Voice '{self.config.voice}' not found, using default"
                            )
                    else:
                        self.logger.warning("No voices available")

                # Set volume
                self.pyttsx3_engine.setProperty("volume", 1.0)

            self.logger.info("pyttsx3 engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pyttsx3: {e}")
            self.pyttsx3_engine = None
            raise

    async def _initialize_coqui(self) -> None:
        """Initialize Coqui TTS with high-quality offline neural voices."""
        try:
            self.logger.info("Initializing Coqui TTS with neural voices...")

            # Initialize in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()

            # Use a fast, high-quality model that works offline
            model_name = (
                "tts_models/en/ljspeech/tacotron2-DDC"  # Good quality, reasonable speed
            )

            self.coqui_tts = await loop.run_in_executor(
                None, lambda: TTS(model_name=model_name)
            )

            self.logger.info(
                "Coqui TTS initialized successfully with neural voice model"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Coqui TTS: {e}")
            self.coqui_tts = None
            raise

    async def _initialize_espeak(self) -> None:
        """Initialize eSpeak-NG (fast, lightweight offline TTS)."""
        try:
            self.logger.info("Initializing eSpeak-NG...")
            # eSpeak-NG doesn't require initialization, just check availability
            result = subprocess.run(
                ["espeak-ng", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.logger.info("eSpeak-NG initialized successfully")
            else:
                raise RuntimeError("eSpeak-NG not working properly")
        except Exception as e:
            self.logger.error(f"Failed to initialize eSpeak-NG: {e}")
            raise

    async def _initialize_bark(self) -> None:
        """Initialize Bark TTS with high-quality neural voices."""
        try:
            self.logger.info("Initializing Bark TTS with neural voices...")

            # Preload models in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()

            def safe_preload():
                """Preload models with PyTorch 2.6 compatibility fix."""
                # PyTorch 2.6 changed the default for weights_only from False to True
                # which breaks loading of Bark models. We temporarily patch torch.load
                # to use weights_only=False for this specific case.
                if torch is None:
                    raise RuntimeError("PyTorch is not available")

                original_torch_load = torch.load

                def patched_torch_load(
                    f, map_location=None, pickle_module=None, **pickle_load_args
                ):
                    # Force weights_only=False for bark model loading
                    if "weights_only" not in pickle_load_args:
                        pickle_load_args["weights_only"] = False
                    return original_torch_load(
                        f,
                        map_location=map_location,
                        pickle_module=pickle_module,
                        **pickle_load_args,
                    )

                try:
                    # Temporarily replace torch.load
                    torch.load = patched_torch_load
                    preload_models()
                finally:
                    # Always restore original torch.load
                    torch.load = original_torch_load

            await loop.run_in_executor(None, safe_preload)

            self.bark_models_loaded = True
            self.logger.info(
                "Bark TTS initialized successfully with neural voice models"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Bark TTS: {e}")
            self.bark_models_loaded = False
            raise

    async def speak(self, text: str) -> None:
        self._emit_state("active", "speaking")
        """
        Convert text to speech and play it.

        Args:
            text: Text to convert to speech
        """
        if not self.is_initialized:
            await self.initialize()

        if not text.strip():
            return

        self.logger.info(f"Speaking: {text}")

        if self.current_backend == "bark":
            await self._speak_bark(text)
        elif self.current_backend == "coqui":
            await self._speak_coqui(text)
        elif self.current_backend == "espeak":
            await self._speak_espeak(text)
        elif self.current_backend == "pyttsx3":
            await self._speak_pyttsx3(text)
        else:
            self.logger.error(
                f"No TTS backend available for speech (current: {self.current_backend})"
            )
            self._emit_state("error", "no backend")
        # Reset interruption flag after each speak invocation (so next response can play)
        self._interrupt_requested = False

    async def _speak_pyttsx3(self, text: str) -> None:
        """
        Generate speech with pyttsx3 by saving to a temporary WAV, then stream
        through AudioManager for unified interruption handling (barge‑in).
        """
        if not self.pyttsx3_engine:
            self.logger.error("pyttsx3 engine not available")
            return

        temp_file = Path("/tmp/pyttsx3_tts_output.wav")
        try:
            if self.audio_manager:
                self.audio_manager.set_speaking_state(True)

            # Synthesize to file in executor (blocking operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: (
                    self.pyttsx3_engine.save_to_file(text, str(temp_file)),
                    self.pyttsx3_engine.runAndWait(),
                ),
            )

            if not temp_file.exists():
                self.logger.error("pyttsx3 did not produce output file")
                return

            # Load audio
            audio_data = await self._load_audio_file(temp_file)
            if audio_data.size == 0:
                self.logger.error("pyttsx3 produced empty audio")
                return

            # Stream via AudioManager with chunk‑level interruption polling
            if self.audio_manager:
                await self.audio_manager.play_audio(
                    audio_data, sample_rate=self.sample_rate
                )
            else:
                # Fallback simulate
                duration = len(audio_data) / self.sample_rate
                await asyncio.sleep(duration)

        except Exception as e:
            self.logger.error(f"pyttsx3 synthesis/playback error: {e}")
        finally:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
            # Minimal post‑playback cooldown (short, non‑blocking for barge‑in)
            await asyncio.sleep(0.15)
            if self.audio_manager:
                self.audio_manager.clear_input_buffer()
                self.audio_manager.set_speaking_state(False)
            self._interrupt_requested = False

    async def _speak_bark(self, text: str) -> None:
        """Generate and play speech using Bark TTS."""
        if not self.bark_models_loaded:
            self.logger.error("Bark TTS models not loaded")
            return

        try:
            # Set speaking state to prevent audio feedback
            if self.audio_manager:
                self.audio_manager.set_speaking_state(True)

            # Generate high-quality neural speech
            loop = asyncio.get_event_loop()

            # Generate audio in separate thread (Bark can be slow)
            temp_file = Path("/tmp/bark_tts_output.wav")

            self.logger.debug("Generating speech with Bark TTS...")

            # Use configured Bark voice preset (history prompt) if provided for deterministic voice
            if getattr(self.config, "bark_voice_preset", None):
                audio_array = await loop.run_in_executor(
                    None,
                    lambda: generate_audio(
                        text, history_prompt=self.config.bark_voice_preset
                    ),
                )
            else:
                audio_array = await loop.run_in_executor(
                    None, lambda: generate_audio(text)
                )

            # Bark returns float32 in range [-1.0, 1.0]; Python's wave module (used in _load_audio_file)
            # only supports PCM (format code 1) and rejects IEEE float WAV (format code 3), which caused:
            # "Error loading audio file: unknown format: 3"
            # Convert to 16-bit PCM before writing to ensure compatibility.
            try:
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array, dtype=np.float32)
                # Clip to avoid overflow, then scale
                if audio_array.dtype != np.int16:
                    audio_pcm16 = np.clip(audio_array, -1.0, 1.0)
                    audio_pcm16 = (audio_pcm16 * 32767.0).astype(np.int16)
                else:
                    audio_pcm16 = audio_array
            except Exception as conv_err:
                self.logger.error(
                    f"Failed converting Bark audio to int16 PCM: {conv_err}"
                )
                return

            # Save audio as PCM16 WAV
            await loop.run_in_executor(
                None, lambda: wavfile.write(str(temp_file), SAMPLE_RATE, audio_pcm16)
            )

            # Load and play the audio
            if temp_file.exists():
                if self.audio_manager:
                    # Load audio data
                    audio_data = await self._load_audio_file(temp_file)
                    if len(audio_data) > 0:
                        await self.audio_manager.play_audio(
                            audio_data, sample_rate=SAMPLE_RATE
                        )
                else:
                    # Fallback: play with system command
                    try:
                        subprocess.run(
                            ["aplay", str(temp_file)], check=True, capture_output=True
                        )
                    except subprocess.CalledProcessError:
                        try:
                            subprocess.run(
                                ["paplay", str(temp_file)],
                                check=True,
                                capture_output=True,
                            )
                        except subprocess.CalledProcessError:
                            self.logger.warning(
                                "Could not play audio with aplay or paplay"
                            )

                # Clean up
                temp_file.unlink()

            # Adaptive minimal pacing pause instead of large heuristic delay.
            # We already know actual audio length; we only need to yield briefly
            # to let playback buffer drain before re‑enabling input.
            actual_duration = max(0.0, len(audio_pcm16) / SAMPLE_RATE)
            pacing_pause = min(
                actual_duration * 0.10, 0.75
            )  # 10% of length, capped at 750ms
            await asyncio.sleep(pacing_pause)

        except Exception as e:
            self.logger.error(f"Bark TTS synthesis error: {e}")
        finally:
            # Configurable short cooldown (replaces fixed 2.0s sleep)
            await asyncio.sleep(self.config.post_tts_cooldown)

            if self.audio_manager:
                self.audio_manager.clear_input_buffer()
                if self.config.enable_tts_buffer_double_clear:
                    await asyncio.sleep(0.12)
                    self.audio_manager.clear_input_buffer()
                self.audio_manager.set_speaking_state(False)
                self.logger.debug(
                    "Speaking state cleared - audio input re-enabled after Bark TTS"
                )
            self._interrupt_requested = False

    async def _speak_coqui(self, text: str) -> None:
        """Generate and play speech using Coqui TTS."""
        if not self.coqui_tts:
            self.logger.error("Coqui TTS not available")
            return

        try:
            # Set speaking state to prevent audio feedback
            if self.audio_manager:
                self.audio_manager.set_speaking_state(True)

            # Generate high-quality speech
            loop = asyncio.get_event_loop()

            # Generate audio in separate thread
            temp_file = Path("/tmp/coqui_tts_output.wav")
            await loop.run_in_executor(
                None,
                lambda: self.coqui_tts.tts_to_file(text=text, file_path=str(temp_file)),
            )

            # Load and play the audio
            if temp_file.exists():
                if self.audio_manager:
                    # Load audio data
                    audio_data = await self._load_audio_file(temp_file)
                    if len(audio_data) > 0:
                        await self.audio_manager.play_audio(
                            audio_data, sample_rate=22050
                        )
                else:
                    # Fallback: play with system command
                    try:
                        subprocess.run(
                            ["aplay", str(temp_file)], check=True, capture_output=True
                        )
                    except subprocess.CalledProcessError:
                        try:
                            subprocess.run(
                                ["paplay", str(temp_file)],
                                check=True,
                                capture_output=True,
                            )
                        except subprocess.CalledProcessError:
                            self.logger.warning(
                                "Could not play audio with aplay or paplay"
                            )

                # Clean up
                temp_file.unlink()

            # Add delay for natural speech timing
            word_count = len(text.split())
            estimated_duration = max(
                2.0, word_count / 2.5
            )  # Neural TTS is typically slower/more natural
            await asyncio.sleep(estimated_duration)

        except Exception as e:
            self.logger.error(f"Coqui TTS synthesis error: {e}")
        finally:
            # Add longer delay before re-enabling audio input
            await asyncio.sleep(2.0)

            if self.audio_manager:
                self.audio_manager.clear_input_buffer()
                await asyncio.sleep(0.5)
                self.audio_manager.clear_input_buffer()
                self.audio_manager.set_speaking_state(False)
                self.logger.debug(
                    "Speaking state cleared - audio input re-enabled after Coqui TTS"
                )
            self._interrupt_requested = False

    async def _speak_espeak(self, text: str) -> None:
        """Generate and play speech using eSpeak-NG."""
        try:
            # Set speaking state to prevent audio feedback
            if self.audio_manager:
                self.audio_manager.set_speaking_state(True)

            # Generate speech with eSpeak-NG
            loop = asyncio.get_event_loop()

            # Use eSpeak-NG with better voice settings
            temp_file = Path("/tmp/espeak_output.wav")

            # eSpeak-NG command with improved voice quality
            espeak_cmd = [
                "espeak-ng",
                "-v",
                "en+f3",  # Female voice variant 3 (more natural)
                "-s",
                "160",  # Speed: 160 words per minute (more natural)
                "-p",
                "50",  # Pitch: 50 (neutral)
                "-a",
                "100",  # Amplitude: 100 (normal volume)
                "-g",
                "10",  # Gap between words: 10ms
                "-w",
                str(temp_file),  # Write to WAV file
                text,
            ]

            # Run eSpeak-NG
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(espeak_cmd, check=True, capture_output=True),
            )

            # Load and play the audio
            if temp_file.exists():
                if self.audio_manager:
                    # Load audio data
                    audio_data = await self._load_audio_file(temp_file)
                    if len(audio_data) > 0:
                        await self.audio_manager.play_audio(
                            audio_data, sample_rate=22050
                        )
                else:
                    # Fallback: play with system command
                    try:
                        subprocess.run(
                            ["aplay", str(temp_file)], check=True, capture_output=True
                        )
                    except subprocess.CalledProcessError:
                        try:
                            subprocess.run(
                                ["paplay", str(temp_file)],
                                check=True,
                                capture_output=True,
                            )
                        except subprocess.CalledProcessError:
                            self.logger.warning(
                                "Could not play audio with aplay or paplay"
                            )

                # Clean up
                temp_file.unlink()

            # Add delay for speech timing
            word_count = len(text.split())
            estimated_duration = max(2.0, word_count / 2.7)  # eSpeak is relatively fast
            await asyncio.sleep(estimated_duration)

        except Exception as e:
            self.logger.error(f"eSpeak TTS synthesis error: {e}")
        finally:
            # Add longer delay before re-enabling audio input
            await asyncio.sleep(2.0)

            if self.audio_manager:
                self.audio_manager.clear_input_buffer()
                await asyncio.sleep(0.5)
                self.audio_manager.clear_input_buffer()
                self.audio_manager.set_speaking_state(False)
                self.logger.debug(
                    "Speaking state cleared - audio input re-enabled after eSpeak TTS"
                )
            self._interrupt_requested = False

    async def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize text to audio data without playing.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array, or None if failed
        """
        if not self.is_initialized:
            await self.initialize()

        if not text.strip():
            return None

        if self.current_backend == "pyttsx3":
            return await self._synthesize_pyttsx3(text)
        elif self.current_backend == "coqui":
            return await self._synthesize_coqui(text)
        elif self.current_backend == "espeak":
            return await self._synthesize_espeak(text)
        else:
            self.logger.error("No TTS backend available for synthesis")
            return None

    async def _synthesize_pyttsx3(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using pyttsx3 (save to file first)."""
        if not self.pyttsx3_engine:
            return None

        try:
            # Save to temporary file
            temp_file = Path("/tmp/tts_output.wav")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: (
                    self.pyttsx3_engine.save_to_file(text, str(temp_file)),
                    self.pyttsx3_engine.runAndWait(),
                ),
            )

            # Load the audio file
            if temp_file.exists():
                audio_data = await self._load_audio_file(temp_file)
                temp_file.unlink()  # Clean up
                return audio_data

            return None

        except Exception as e:
            self.logger.error(f"pyttsx3 synthesis error: {e}")
            return None

    async def _load_audio_file(self, file_path: Path) -> np.ndarray:
        """Load audio data from file."""
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)
                return audio_data
        except Exception as e:
            self.logger.error(f"Error loading audio file: {e}")
            return np.array([])

    async def _synthesize_coqui(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Coqui TTS (returns audio data)."""
        if not self.coqui_tts:
            return None

        try:
            temp_file = Path("/tmp/coqui_synthesis.wav")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.coqui_tts.tts_to_file(text=text, file_path=str(temp_file)),
            )

            if temp_file.exists():
                audio_data = await self._load_audio_file(temp_file)
                temp_file.unlink()
                return audio_data

            return None

        except Exception as e:
            self.logger.error(f"Coqui synthesis error: {e}")
            return None

    async def _synthesize_espeak(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using eSpeak-NG (returns audio data)."""
        try:
            temp_file = Path("/tmp/espeak_synthesis.wav")

            espeak_cmd = [
                "espeak-ng",
                "-v",
                "en+f3",
                "-s",
                "160",
                "-p",
                "50",
                "-a",
                "100",
                "-g",
                "10",
                "-w",
                str(temp_file),
                text,
            ]

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(espeak_cmd, check=True, capture_output=True),
            )

            if temp_file.exists():
                audio_data = await self._load_audio_file(temp_file)
                temp_file.unlink()
                return audio_data

            return None

        except Exception as e:
            self.logger.error(f"eSpeak synthesis error: {e}")
            return None

    async def _synthesize_bark(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Bark TTS (returns audio data)."""
        if not self.bark_models_loaded:
            return None

        try:
            # Generate audio in separate thread
            loop = asyncio.get_event_loop()

            if getattr(self.config, "bark_voice_preset", None):
                audio_array = await loop.run_in_executor(
                    None,
                    lambda: generate_audio(
                        text, history_prompt=self.config.bark_voice_preset
                    ),
                )
            else:
                audio_array = await loop.run_in_executor(
                    None, lambda: generate_audio(text)
                )

            # Convert to numpy array with correct format
            # Bark returns float32, but we need int16 for compatibility
            if isinstance(audio_array, np.ndarray):
                if audio_array.dtype != np.int16:
                    # Normalize and convert to int16
                    audio_array = (audio_array * 32767).astype(np.int16)
                return audio_array
            else:
                # Convert to numpy array first
                audio_array = np.array(audio_array)
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)
                return audio_array

        except Exception as e:
            self.logger.error(f"Bark synthesis error: {e}")
            return None

    async def _play_audio_array(self, audio_data: np.ndarray) -> None:
        """
        Play audio data through the audio manager.

        Args:
            audio_data: Audio data to play
        """
        if self.audio_manager:
            try:
                # AudioManager's play_audio method will handle speaking state management
                # Pass the sample rate so AudioManager can handle resampling if needed
                await self.audio_manager.play_audio(
                    audio_data, sample_rate=self.sample_rate
                )
                self.logger.debug("Audio playback completed through AudioManager")
            except Exception as e:
                self.logger.error(f"Error playing audio through audio manager: {e}")
                # Clear speaking state on error
                if self.audio_manager:
                    self.audio_manager.set_speaking_state(False)
            finally:
                # Ensure speaking state is cleared
                if self.audio_manager:
                    self.audio_manager.set_speaking_state(False)
        else:
            # Fallback: simulate playback time
            self.logger.debug(
                f"No audio manager available, simulating playback of {len(audio_data)} samples"
            )
            duration = len(audio_data) / self.sample_rate
            await asyncio.sleep(duration)

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices.

        Returns:
            List of voice information dictionaries
        """
        voices = []

        if self.current_backend == "pyttsx3" and self.pyttsx3_engine:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty("voices")
                for voice in pyttsx3_voices:
                    voices.append(
                        {
                            "id": voice.id,
                            "name": voice.name,
                            "gender": getattr(voice, "gender", "unknown"),
                            "age": getattr(voice, "age", "unknown"),
                            "backend": "pyttsx3",
                        }
                    )
            except Exception as e:
                self.logger.error(f"Error getting pyttsx3 voices: {e}")

        return voices

    def set_voice(self, voice_id: str) -> bool:
        """
        Set the current voice.

        Args:
            voice_id: ID of the voice to use

        Returns:
            True if voice was set successfully
        """
        try:
            if self.current_backend == "pyttsx3" and self.pyttsx3_engine:
                self.pyttsx3_engine.setProperty("voice", voice_id)
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error setting voice: {e}")
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the TTS service.

        Returns:
            Dictionary containing service information
        """
        return {
            "backend": self.current_backend,
            "engine": self.config.engine,
            "voice": self.config.voice,
            "speed": self.config.speed,
            "sample_rate": self.sample_rate,
            "initialized": self.is_initialized,
        }

    async def cleanup(self) -> None:
        """Cleanup TTS resources."""
        self.logger.info("Cleaning up TTS service...")

        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping pyttsx3 engine: {e}")
            self.pyttsx3_engine = None

        self.is_initialized = False

        self.logger.info("TTS service cleanup complete")
