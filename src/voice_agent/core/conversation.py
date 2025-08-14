"""
Main conversation manager and VoiceAgent class.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audio_manager import AudioManager
from .config import Config
from .llm_service import LLMService
from .stt_service import STTService
from .tool_executor import ToolExecutor
from .tts_service import TTSService


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
        self, config: Optional[Config] = None, config_path: Optional[Path] = None
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
        elif config_path:
            self.config = Config.load(config_path)
        else:
            # Use default config
            default_config_path = (
                Path(__file__).parent.parent / "config" / "default.yaml"
            )
            self.config = Config.load(default_config_path)

        # Initialize components
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

        # Initialize audio manager
        self.audio_manager = AudioManager(self.config.audio)
        await self.audio_manager.initialize()

        # Initialize STT service
        self.stt_service = STTService(self.config.stt)
        await self.stt_service.initialize()

        # Initialize TTS service (pass audio manager for playback integration)
        self.tts_service = TTSService(self.config.tts, self.audio_manager)
        await self.tts_service.initialize()

        # Initialize LLM service
        self.llm_service = LLMService(self.config.llm)
        await self.llm_service.initialize()

        # Initialize tool executor
        self.tool_executor = ToolExecutor(self.config.tools)
        await self.tool_executor.initialize()

        self.logger.info("Voice Agent initialized successfully")

    async def start(self) -> None:
        """Start the voice agent main loop."""
        if not all(
            [
                self.audio_manager,
                self.stt_service,
                self.tts_service,
                self.llm_service,
                self.tool_executor,
            ]
        ):
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
        """Main conversation loop."""
        while self.is_running:
            try:
                # Listen for user input
                audio_data = await self.audio_manager.listen()
                if audio_data is None or (
                    hasattr(audio_data, "size") and audio_data.size == 0
                ):
                    continue

                # Convert speech to text
                user_input = await self.stt_service.transcribe(audio_data)
                if not user_input.strip():
                    continue

                self.logger.info(f"User: {user_input}")

                # Process with LLM
                response = await self.llm_service.generate_response(
                    user_input, self.conversation_history, self.tool_executor
                )

                self.logger.info(f"Agent: {response}")

                # Convert text to speech and play
                await self.tts_service.speak(response)

                # Update conversation history
                self._update_history(user_input, response)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

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

    async def process_text(self, text: str) -> str:
        """
        Process text input without audio (for testing/debugging).

        Args:
            text: Input text to process

        Returns:
            Agent response text
        """
        if not self.llm_service:
            await self.initialize()

        response = await self.llm_service.generate_response(
            text, self.conversation_history, self.tool_executor
        )

        self._update_history(text, response)
        return response
