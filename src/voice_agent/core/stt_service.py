"""
Speech-to-Text service for the voice agent.
"""

import asyncio
import logging
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    def __init__(self, config: STTConfig):
        """
        Initialize the STT service.

        Args:
            config: STT configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

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

    async def initialize(self) -> None:
        """Initialize the STT service and load models."""
        self.logger.info(
            f"Initializing STT service with backend: {self.current_backend}"
        )

        if self.current_backend == "whisper":
            await self._initialize_whisper()
        elif self.current_backend == "vosk":
            await self._initialize_vosk()
        else:
            self.logger.error("No STT backend could be initialized")
            return

        self.is_initialized = True
        self.logger.info("STT service initialized")

    async def _initialize_whisper(self) -> None:
        """Initialize Faster Whisper model."""
        try:
            model_name = self.config.model
            if model_name.startswith("whisper-"):
                model_name = model_name.replace("whisper-", "")

            self.logger.info(f"Loading Whisper model: {model_name}")

            # Load model in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    model_name,
                    device="auto",
                    compute_type="int8",  # Use int8 for better performance
                ),
            )

            self.logger.info(f"Whisper model '{model_name}' loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
            raise

    async def _initialize_vosk(self) -> None:
        """Initialize Vosk model."""
        try:
            model_path = self._get_vosk_model_path()

            if not model_path.exists():
                self.logger.error(f"Vosk model not found at {model_path}")
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
            self.logger.error(f"Failed to load Vosk model: {e}")
            self.vosk_model = None
            self.vosk_recognizer = None

    def _get_vosk_model_path(self) -> Path:
        """Get the path to the Vosk model."""
        # This should be configurable or downloaded automatically
        model_dir = Path.home() / ".cache" / "voice_agent" / "vosk_models"
        return model_dir / self.config.model

    async def transcribe(self, audio_data: np.ndarray) -> str:
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
            return await self._transcribe_whisper(audio_data)
        elif self.current_backend == "vosk":
            return await self._transcribe_vosk(audio_data)
        else:
            self.logger.error("No STT backend available for transcription")
            return ""

    async def _transcribe_whisper(self, audio_data: np.ndarray) -> str:
        """Transcribe using Faster Whisper."""
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

            # Transcribe in a separate thread
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.whisper_model.transcribe(
                    audio_float,
                    language=(
                        self.config.language if self.config.language != "auto" else None
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
                self.logger.info(f"Transcribed: {text}")
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
