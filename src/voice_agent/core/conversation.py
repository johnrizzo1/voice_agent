"""
Main conversation manager and VoiceAgent class.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .audio_manager import AudioManager
from .config import Config
from .llm_service import LLMService
from .stt_service import STTService
from .tool_executor import ToolExecutor
from .tts_service import TTSService


StateCallback = Callable[[str, str, Optional[str]], None]


class VoiceAgent:
    """
    Main Voice Agent class that orchestrates all components.

    This class manages the complete voice interaction pipeline:
    - Audio input/output
    - Speech-to-text conversion
    - Language model processing
    - Tool execution
    - Text-to-speech synthesis
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[Path] = None,
        text_only: bool = False,
        state_callback: Optional["StateCallback"] = None,
    ):
        """
        Initialize the Voice Agent.

        Args:
            config: Configuration object
            config_path: Path to configuration file (if config not provided)
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config:
            self.config = config
            self._config_path: Optional[Path] = None
        elif config_path:
            self.config = Config.load(config_path)
            self._config_path = config_path
        else:
            # Use default config (remember path for persistence of UI toggles)
            default_config_path = (
                Path(__file__).parent.parent / "config" / "default.yaml"
            )
            self.config = Config.load(default_config_path)
            self._config_path = default_config_path

        # Mode flags
        ui_cfg = getattr(self.config, "ui", None)
        force_text_only_cfg = (
            getattr(ui_cfg, "force_text_only", False) if ui_cfg else False
        )
        self.text_only = text_only or force_text_only_cfg

        # Optional pipeline state callback (component, state, message)
        self._state_callback = state_callback

        # Initialize component placeholders
        self.audio_manager: Optional[AudioManager] = None
        self.stt_service: Optional[STTService] = None
        self.tts_service: Optional[TTSService] = None
        self.llm_service: Optional[LLMService] = None
        self.tool_executor: Optional[ToolExecutor] = None

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_running = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        self.logger.info("Initializing Voice Agent components...")

        if not self.text_only:
            # Initialize audio manager
            self.audio_manager = AudioManager(
                self.config.audio, state_callback=self._state_callback
            )
            await self.audio_manager.initialize()

            # Initialize STT service
            self.stt_service = STTService(
                self.config.stt, state_callback=self._state_callback
            )
            await self.stt_service.initialize()

            # Initialize TTS service (pass audio manager for playback integration)
            self.tts_service = TTSService(
                self.config.tts, self.audio_manager, state_callback=self._state_callback
            )
            await self.tts_service.initialize()

            # Wire barge-in / interruption between audio manager & TTS
            try:
                if self.audio_manager and self.tts_service:
                    # When user speech detected during TTS, request interruption
                    self.audio_manager.set_barge_in_callback(
                        lambda: self.tts_service.request_interrupt()
                    )
                    # Audio playback checks this predicate each chunk
                    self.audio_manager.set_interrupt_getter(
                        lambda: bool(
                            getattr(self.tts_service, "_interrupt_requested", False)
                        )
                    )
            except Exception:
                self.logger.debug("Failed wiring barge-in callbacks", exc_info=True)
        else:
            self.logger.info(
                "Text-only mode: skipping audio / STT / TTS initialization"
            )

        # Initialize LLM service
        self.llm_service = LLMService(
            self.config.llm, state_callback=self._state_callback
        )
        await self.llm_service.initialize()

        # Initialize tool executor
        self.tool_executor = ToolExecutor(self.config.tools)
        await self.tool_executor.initialize()

        self.logger.info("Voice Agent initialized successfully")

    async def start(self) -> None:
        """Start the voice agent main loop."""
        required = [self.llm_service, self.tool_executor]
        if not self.text_only:
            required.extend([self.audio_manager, self.stt_service, self.tts_service])

        if not all(required):
            await self.initialize()

        self.is_running = True
        self.logger.info("Voice Agent started")

        try:
            await self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Voice Agent stopped by user")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the voice agent and cleanup resources."""
        self.is_running = False
        self.logger.info("Stopping Voice Agent...")

        if self.audio_manager:
            await self.audio_manager.cleanup()
        if self.stt_service:
            await self.stt_service.cleanup()
        if self.tts_service:
            await self.tts_service.cleanup()
        if self.llm_service:
            await self.llm_service.cleanup()
        if self.tool_executor:
            await self.tool_executor.cleanup()

        self.logger.info("Voice Agent stopped")

    async def _main_loop(self) -> None:
        """Main conversation loop.

        In text_only mode this loop idles (input is provided via external calls,
        e.g. TUI adapter invoking process_text). In voice mode it performs the
        full capture → STT → LLM → TTS pipeline.
        """
        while self.is_running:
            try:
                if self.text_only:
                    # Idle briefly; external components (TUI) drive interactions.
                    await asyncio.sleep(0.25)
                    continue

                # Ensure required components are initialized (runtime safety + type narrowing)
                if not all(
                    [
                        self.audio_manager,
                        self.stt_service,
                        self.tts_service,
                        self.llm_service,
                        self.tool_executor,
                    ]
                ):
                    self.logger.warning(
                        "One or more pipeline components missing (reinitializing)..."
                    )
                    await self.initialize()
                    await asyncio.sleep(0.25)
                    continue

                # Type narrowing for static analyzers
                assert (
                    self.audio_manager
                    and self.stt_service
                    and self.tts_service
                    and self.llm_service
                    and self.tool_executor is not None
                )

                # Voice mode: capture microphone audio
                audio_data = await self.audio_manager.listen()
                if audio_data is None or (
                    hasattr(audio_data, "size") and audio_data.size == 0
                ):
                    continue

                # STT
                user_input = await self.stt_service.transcribe(audio_data)
                if not user_input.strip():
                    continue
                self.logger.info(f"User: {user_input}")

                # LLM
                response = await self.llm_service.generate_response(
                    user_input, self.conversation_history, self.tool_executor
                )
                self.logger.info(f"Agent: {response}")

                # TTS
                await self.tts_service.speak(response)

                # History
                self._update_history(user_input, response)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    def _update_history(self, user_input: str, agent_response: str) -> None:
        """Update conversation history."""
        self.conversation_history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": agent_response},
            ]
        )

        # Keep history within limits
        max_history = self.config.conversation.max_history
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def set_state_callback(self, callback: Optional["StateCallback"]) -> None:
        """Install or replace the pipeline state callback and propagate to components."""
        self._state_callback = callback
        if self.audio_manager:
            self.audio_manager.set_state_callback(callback)
        if self.stt_service:
            self.stt_service.set_state_callback(callback)
        if self.tts_service:
            self.tts_service.set_state_callback(callback)
        if self.llm_service:
            self.llm_service.set_state_callback(callback)

    async def process_text(self, text: str) -> str:
        """
        Process text input without audio (for testing/debugging).

        Args:
            text: Input text to process

        Returns:
            Agent response text
        """
        if not self.llm_service or not self.tool_executor:
            await self.initialize()
            if not self.llm_service:
                raise RuntimeError("LLM service failed to initialize")
        # Narrow types for static analyzers
        assert self.llm_service is not None
        assert self.tool_executor is not None

        response = await self.llm_service.generate_response(
            text, self.conversation_history, self.tool_executor
        )

        self._update_history(text, response)
        return response
