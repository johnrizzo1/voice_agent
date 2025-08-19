"""
Speech-to-Text service for the voice agent.
"""

import asyncio
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from .config import STTConfig

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import json

    import vosk
except ImportError:
    vosk = None


class STTService:
    """
    Speech-to-Text service supporting multiple backends.

    Supports:
    - Faster Whisper (primary, high accuracy)
    - Vosk (lightweight, streaming)
    """

    def __init__(
        self,
        config: STTConfig,
        state_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
    ):
        """
        Initialize the STT service.

        Args:
            config: STT configuration settings
            state_callback: Optional callback(component, state, message)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._state_callback = state_callback

        # Model instances
        self.whisper_model: Optional[WhisperModel] = None
        self.vosk_model: Optional[vosk.Model] = None
        self.vosk_recognizer: Optional[vosk.KaldiRecognizer] = None

        # Service state
        self.is_initialized = False
        self.current_backend = self._determine_backend()

    def _determine_backend(self) -> str:
        """Determine which STT backend to use based on availability."""
        if "whisper" in self.config.model.lower() and WhisperModel:
            return "whisper"
        elif "vosk" in self.config.model.lower() and vosk:
            return "vosk"
        elif WhisperModel:
            return "whisper"
        elif vosk:
            return "vosk"
        else:
            self.logger.error("No STT backend available")
            return "none"

    def set_state_callback(
        self, cb: Optional[Callable[[str, str, Optional[str]], None]]
    ) -> None:
        """Set/replace state callback."""
        self._state_callback = cb

    def _emit_state(self, state: str, message: Optional[str] = None) -> None:
        if self._state_callback:
            try:
                self._state_callback("stt", state, message)
            except Exception:
                self.logger.debug("STT state callback error", exc_info=True)

    async def initialize(self) -> None:
        """
        Initialize the STT service and load models with layered fallback:
          1. Preferred backend (config / availability)
          2. Alternate backend (Whisper <-> Vosk)
          3. Dummy backend (only if allow_dummy == True)

        Emits 'ready' even for dummy backend (instead of permanent 'error') so the
        UI does not remain stuck, unless no fallback is allowed.
        """
        if self.is_initialized:
            return

        self.logger.info(
            f"Initializing STT service (preferred backend: {self.current_backend})"
        )
        self._emit_state("initializing", f"backend={self.current_backend}")

        # Helper to mark ready
        def mark_ready(msg: Optional[str] = None):
            self.is_initialized = True
            self.logger.info(f"STT service initialized ({msg or self.current_backend})")
            self._emit_state("ready", msg)

        # Attempt preferred backend
        try_order: List[str] = []
        if self.current_backend == "whisper":
            try_order = ["whisper", "vosk"]
        elif self.current_backend == "vosk":
            try_order = ["vosk", "whisper"]
        else:
            # Unknown / none – try both
            try_order = ["whisper", "vosk"]

        backend_success = False
        for backend in try_order:
            try:
                if backend == "whisper":
                    self.logger.debug("Attempting Whisper initialization...")
                    await self._initialize_whisper()
                    if getattr(self, "model_type", None) in (
                        "faster_whisper",
                        "whisper",
                    ):
                        self.current_backend = "whisper"
                        backend_success = True
                        break
                    # If model_type became 'dummy' here, treat as failure and continue
                elif backend == "vosk":
                    if vosk:
                        self.logger.debug("Attempting Vosk initialization...")
                        await self._initialize_vosk()
                        if self.vosk_recognizer:
                            self.current_backend = "vosk"
                            backend_success = True
                            break
                    else:
                        self.logger.debug("Vosk not installed; skipping")
            except Exception as be:
                self.logger.warning(
                    f"{backend} initialization failed: {be}", exc_info=True
                )
                continue

        if backend_success:
            mark_ready()
            return

        # If we reach here, neither backend succeeded. Initialize dummy backend.
        allow_dummy = self.config.allow_dummy_fallback
        if allow_dummy:
            self.logger.warning(
                "All STT backends failed. Activating dummy STT backend (placeholder transcriptions)."
            )
            self.whisper_model = None
            self.vosk_model = None
            self.vosk_recognizer = None
            self.model_type = "dummy"
            self.current_backend = "dummy"
            mark_ready("dummy-backend")
        else:
            self._emit_state("error", "no backend available")
            self.logger.error("STT initialization failed with no permitted fallback")

    async def _initialize_whisper(self) -> None:
        """Initialize Faster Whisper model."""
        import os
        import platform

        self.logger.debug(f"WhisperModel available: {WhisperModel is not None}")

        if WhisperModel is None:
            raise ImportError("faster-whisper package not available")

        try:
            model_name = self.config.model
            if model_name.startswith("whisper-"):
                model_name = model_name.replace("whisper-", "")

            self.logger.info(f"Loading Whisper model: {model_name}")

            # Handle macOS-specific issues with faster-whisper
            whisper_kwargs = {
                "device": "auto",
                "compute_type": "int8",  # Use int8 for better performance
            }

            # Apply macOS-specific workarounds
            if platform.system() == "Darwin":
                self.logger.debug("Applying macOS compatibility settings for Whisper")
                # Force CPU mode on macOS to avoid PortAudio/CoreAudio conflicts
                whisper_kwargs["device"] = "cpu"
                # Set environment variables to avoid file descriptor issues
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

                # On macOS, use standard whisper instead of faster-whisper to avoid fd issues
                try:
                    self.logger.info(
                        f"Loading standard Whisper model on macOS: {model_name}"
                    )
                    import whisper

                    loop = asyncio.get_event_loop()
                    self.whisper_model = await loop.run_in_executor(
                        None,
                        lambda: whisper.load_model(model_name, device="cpu"),
                    )
                    self.model_type = "whisper"  # Track which model type we're using
                    self.logger.info(
                        "Standard Whisper model loaded successfully on macOS"
                    )
                    return
                except Exception as e:
                    self.logger.error(
                        f"Failed to load standard Whisper model on macOS: {e}"
                    )
                    # Fall through to try faster-whisper as last resort

            # Load model in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(model_name, **whisper_kwargs),
            )
            self.model_type = "faster_whisper"  # Track which model type we're using

            self.logger.info(f"Whisper model '{model_name}' loaded successfully")
        except Exception as e:
            self.logger.warning(
                f"Whisper initialization failed: {type(e).__name__}: {str(e)[:100]}"
            )
            # Try fallback initialization with minimal parameters and process isolation
            try:
                self.logger.info("Attempting fallback Whisper initialization...")
                if platform.system() == "Darwin":
                    # On macOS, try more aggressive isolation
                    self.logger.info("Using process isolation for macOS fallback")
                    loop = asyncio.get_event_loop()
                    self.whisper_model = await loop.run_in_executor(
                        None,
                        lambda: self._create_whisper_with_isolation(model_name),
                    )
                else:
                    # Standard fallback for other platforms
                    loop = asyncio.get_event_loop()
                    self.whisper_model = await loop.run_in_executor(
                        None,
                        lambda: WhisperModel(
                            model_name,
                            device="cpu",
                            compute_type="int8",
                            num_workers=1,
                        ),
                    )
                self.logger.info("Whisper model loaded with fallback settings")
            except Exception as fallback_error:
                self.logger.warning(
                    f"Whisper fallback failed: {type(fallback_error).__name__}: {str(fallback_error)[:100]}"
                )
                # Mark as failed - let main initialize() handle Vosk fallback
                raise fallback_error

    def _create_whisper_with_isolation(self, model_name: str):
        """Create Whisper model with process-level isolation for macOS."""
        import gc
        import os
        import tempfile
        from contextlib import contextmanager

        @contextmanager
        def isolated_environment():
            """Context manager for isolated model creation."""
            # Save original environment
            original_env = {}
            env_vars = ["OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "MKL_NUM_THREADS"]
            for var in env_vars:
                original_env[var] = os.environ.get(var)

            try:
                # Set isolation environment
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ["MKL_NUM_THREADS"] = "1"

                # Force garbage collection
                gc.collect()

                yield

            finally:
                # Restore original environment
                for var, value in original_env.items():
                    if value is not None:
                        os.environ[var] = value
                    elif var in os.environ:
                        del os.environ[var]

        # Try different initialization strategies
        strategies = [
            # Strategy 1: Minimal configuration
            lambda: WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                num_workers=1,
                download_root=tempfile.gettempdir(),
            ),
            # Strategy 2: Auto compute type
            lambda: WhisperModel(
                model_name, device="cpu", compute_type="auto", num_workers=1
            ),
            # Strategy 3: No threading at all
            lambda: WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                num_workers=0,  # Disable threading completely
            ),
        ]

        last_error = None
        for i, strategy in enumerate(strategies, 1):
            try:
                self.logger.debug(
                    f"Trying Whisper isolation strategy {i}/{len(strategies)}"
                )
                with isolated_environment():
                    model = strategy()
                    self.logger.info(
                        f"Whisper model created successfully with strategy {i}"
                    )
                    return model
            except Exception as e:
                last_error = e
                self.logger.debug(f"Strategy {i} failed: {e}")
                continue

        # If all strategies failed, raise the last error
        raise Exception(f"All isolation strategies failed. Last error: {last_error}")

    async def _initialize_vosk(self) -> None:
        """Initialize Vosk model with fallback model name resolution and auto-download."""
        try:
            vosk_model_name = self._resolve_vosk_model_name()
            model_path = self._get_vosk_model_path(vosk_model_name)

            self.logger.debug(
                f"Vosk model path: {model_path} (exists: {model_path.exists()})"
            )

            if not model_path.exists():
                self.logger.warning(
                    f"Vosk model not found at {model_path}. Attempting auto-download..."
                )
                success = await self._download_vosk_model(
                    vosk_model_name, model_path.parent
                )
                if not success:
                    self.logger.error(
                        f"Vosk model download failed. Configure stt.fallback_vosk_model or manually download to {model_path}"
                    )
                    return

            # Load model in a separate thread
            loop = asyncio.get_event_loop()
            self.vosk_model = await loop.run_in_executor(
                None, lambda: vosk.Model(str(model_path))
            )

            # Create recognizer
            self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)

            self.logger.info(f"Vosk model loaded from {model_path}")
        except Exception as e:
            self.logger.warning(
                f"Vosk initialization failed: {type(e).__name__}: {str(e)[:100]}"
            )
            self.vosk_model = None
            self.vosk_recognizer = None

    def _resolve_vosk_model_name(self) -> str:
        """Resolve Vosk model name from config."""
        if "vosk" in self.config.model.lower():
            return self.config.model
        else:
            # Using Whisper model name but need Vosk fallback
            return self.config.fallback_vosk_model

    def _get_vosk_model_path(self, model_name: str) -> Path:
        """Get the path to the Vosk model."""
        model_dir = Path.home() / ".cache" / "voice_agent" / "vosk_models"
        return model_dir / model_name

    async def _download_vosk_model(self, model_name: str, model_dir: Path) -> bool:
        """Download Vosk model automatically."""
        import zipfile
        import tempfile
        from urllib.request import urlretrieve

        # Common Vosk model URLs (expand as needed)
        model_urls = {
            "vosk-model-en-us-0.22": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            "vosk-model-en-us-0.22-lgraph": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip",
            "vosk-model-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        }

        if model_name not in model_urls:
            self.logger.error(
                f"Unknown Vosk model: {model_name}. Available models: {list(model_urls.keys())}"
            )
            return False

        url = model_urls[model_name]
        self.logger.info(f"Downloading Vosk model {model_name} from {url}...")

        try:
            # Create model directory
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: urlretrieve(url, tmp_file.name)
                )

                # Extract zip file
                with zipfile.ZipFile(tmp_file.name, "r") as zip_ref:
                    zip_ref.extractall(model_dir)

                # Clean up temporary file
                Path(tmp_file.name).unlink()

            self.logger.info(f"Vosk model {model_name} downloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download Vosk model: {e}")
            return False

    async def transcribe(self, audio_data: np.ndarray) -> str:
        self._emit_state("active", "transcribing")
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text
        """
        if not self.is_initialized:
            await self.initialize()

        if self.current_backend == "whisper":
            result = await self._transcribe_whisper(audio_data)
        elif self.current_backend == "vosk":
            result = await self._transcribe_vosk(audio_data)
        elif getattr(self, "model_type", None) == "dummy":
            # Dummy backend: keep pipeline healthy (READY not ERROR)
            placeholder = (
                "[Dummy STT: Voice input detected but STT backend unavailable]"
            )
            # Log once-per-session warning about missing models
            if not hasattr(self, "_dummy_warning_logged"):
                self._dummy_warning_logged = True
                self.logger.warning(
                    "STT using dummy backend. Install faster-whisper (pip install faster-whisper) or download Vosk model to enable real transcription"
                )
            self.logger.debug(
                "Dummy STT backend returning placeholder transcription (no error state)"
            )
            self._emit_state("ready", "dummy-backend")
            return placeholder
        else:
            self.logger.error("No STT backend available for transcription")
            self._emit_state("error", "no backend for transcription")
            return ""
        self._emit_state("ready", None)
        return result

    async def _transcribe_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe using Whisper (either standard or faster-whisper)."""
        # Handle dummy STT backend (try dynamic Vosk fallback if now available)
        if hasattr(self, "model_type") and self.model_type == "dummy":
            if vosk:
                try:
                    if not self.vosk_recognizer:
                        self.logger.info(
                            "Attempting late Vosk initialization to replace dummy STT backend..."
                        )
                        await self._initialize_vosk()
                        if self.vosk_recognizer:
                            self.current_backend = "vosk"
                            self.model_type = "vosk"
                            self.logger.info(
                                "Late Vosk fallback initialized successfully"
                            )
                            # Convert to int16 and route to Vosk path
                            audio_int16 = audio_data.astype(np.int16)
                            return await self._transcribe_vosk(audio_int16)
                except Exception as late_fallback_error:
                    self.logger.warning(
                        f"Late Vosk fallback failed: {late_fallback_error}"
                    )
            self.logger.debug("Using dummy STT backend - returning placeholder text")
            return "[Dummy STT: Voice input detected but STT is not available]"

        if not self.whisper_model:
            self.logger.error("Whisper model not available")
            return ""

        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Ensure minimum length for Whisper
            if len(audio_float) < 1600:  # Minimum ~0.1 seconds at 16kHz
                self.logger.debug("Audio too short for transcription")
                return ""

            self.logger.debug(
                f"Transcribing audio of length: {len(audio_float)} samples"
            )

            # Transcribe in a separate thread - handle both model types
            loop = asyncio.get_event_loop()

            # Check if we're using standard whisper (macOS) or faster-whisper
            if hasattr(self, "model_type") and self.model_type == "whisper":
                # Standard whisper returns a dict with 'text' key
                result = await loop.run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(
                        audio_float,
                        language=(
                            self.config.language
                            if self.config.language != "auto"
                            else None
                        ),
                    ),
                )
                text = result.get("text", "").strip()
                self.logger.info(f"Standard Whisper transcribed: {text}")
                return text
            else:
                # Faster-whisper returns segments and info
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(
                        audio_float,
                        language=(
                            self.config.language
                            if self.config.language != "auto"
                            else None
                        ),
                        vad_filter=False,  # Disable VAD filtering for now to test transcription
                    ),
                )

                # Extract text from segments
                text_parts = []
                segment_count = 0
                for segment in segments:
                    segment_count += 1
                    self.logger.debug(
                        f"Segment {segment_count}: '{segment.text}' (confidence: {segment.avg_logprob:.3f})"
                    )
                    if segment.text.strip():
                        text_parts.append(segment.text.strip())

                text = " ".join(text_parts)

                self.logger.info(
                    f"Total segments: {segment_count}, transcribed parts: {len(text_parts)}"
                )

                if text:
                    self.logger.info(f"Faster-Whisper transcribed: {text}")
                else:
                    self.logger.warning(
                        "No speech detected in audio - all segments were empty or low confidence"
                    )

                return text.strip()

        except Exception as e:
            self.logger.error(f"Whisper transcription error: {e}")
            return ""

    async def _transcribe_vosk(self, audio_data: np.ndarray) -> str:
        """Transcribe using Vosk."""
        if not self.vosk_recognizer:
            self.logger.error("Vosk recognizer not available")
            return ""

        try:
            # Convert to bytes
            audio_bytes = audio_data.astype(np.int16).tobytes()

            # Process in chunks for streaming
            loop = asyncio.get_event_loop()

            # Accept audio data
            await loop.run_in_executor(
                None, lambda: self.vosk_recognizer.AcceptWaveform(audio_bytes)
            )

            # Get final result
            result_json = await loop.run_in_executor(
                None, self.vosk_recognizer.FinalResult
            )

            result = json.loads(result_json)
            text = result.get("text", "")

            self.logger.debug(f"Vosk transcription: {text}")
            return text.strip()

        except Exception as e:
            self.logger.error(f"Vosk transcription error: {e}")
            return ""

    async def transcribe_streaming(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Transcribe audio chunk for streaming (partial results).

        Args:
            audio_chunk: Audio chunk as numpy array

        Returns:
            Partial transcription result or None
        """
        if not self.is_initialized:
            await self.initialize()

        if self.current_backend == "vosk":
            return await self._transcribe_streaming_vosk(audio_chunk)
        else:
            # Whisper doesn't support streaming well, fallback to batch
            return None

    async def _transcribe_streaming_vosk(
        self, audio_chunk: np.ndarray
    ) -> Optional[str]:
        """Streaming transcription using Vosk."""
        if not self.vosk_recognizer:
            return None

        try:
            audio_bytes = audio_chunk.astype(np.int16).tobytes()

            loop = asyncio.get_event_loop()
            accepted = await loop.run_in_executor(
                None, lambda: self.vosk_recognizer.AcceptWaveform(audio_bytes)
            )

            if accepted:
                # Final result
                result_json = await loop.run_in_executor(
                    None, self.vosk_recognizer.Result
                )
                result = json.loads(result_json)
                return result.get("text", "")
            else:
                # Partial result
                partial_json = await loop.run_in_executor(
                    None, self.vosk_recognizer.PartialResult
                )
                partial = json.loads(partial_json)
                return partial.get("partial", "")

        except Exception as e:
            self.logger.debug(f"Streaming transcription error: {e}")
            return None

    async def transcribe_file(self, file_path: Path) -> str:
        """
        Transcribe audio from a file.

        Args:
            file_path: Path to audio file

        Returns:
            Transcribed text
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Load audio file
            audio_data = await self._load_audio_file(file_path)
            return await self.transcribe(audio_data)
        except Exception as e:
            self.logger.error(f"File transcription error: {e}")
            return ""

    async def _load_audio_file(self, file_path: Path) -> np.ndarray:
        """Load audio data from file."""
        # This would need proper audio file loading
        # For now, assume it's a WAV file
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)
                return audio_data
        except Exception as e:
            self.logger.error(f"Error loading audio file: {e}")
            return np.array([])

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported models.

        Returns:
            List of supported model names
        """
        models = []

        if WhisperModel:
            models.extend(
                [
                    "whisper-tiny",
                    "whisper-base",
                    "whisper-small",
                    "whisper-medium",
                    "whisper-large",
                ]
            )

        if vosk:
            # This would list available Vosk models
            models.extend(["vosk-model-en-us-0.22"])

        return models

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        return {
            "backend": self.current_backend,
            "model": self.config.model,
            "language": self.config.language,
            "streaming": self.config.streaming,
            "initialized": self.is_initialized,
        }

    async def cleanup(self) -> None:
        """Cleanup STT resources."""
        self.logger.info("Cleaning up STT service...")

        self.whisper_model = None
        self.vosk_model = None
        self.vosk_recognizer = None
        self.is_initialized = False

        self.logger.info("STT service cleanup complete")
