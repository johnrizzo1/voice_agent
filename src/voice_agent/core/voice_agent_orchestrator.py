"""
VoiceAgent Orchestrator with Multi-Agent Integration.

This module provides the main orchestrator class that integrates the existing
voice pipeline with the multi-agent framework while maintaining full backward
compatibility with single-agent mode.
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
from .multi_agent_service import MultiAgentService
from .llamaindex_service import LlamaIndexService


StateCallback = Callable[[str, str, Optional[str]], None]


class VoiceAgentOrchestrator:
    """
    Enhanced Voice Agent orchestrator with multi-agent capabilities.

    This class extends the existing VoiceAgent functionality with intelligent
    multi-agent routing while maintaining full backward compatibility. It acts
    as the main orchestrator for the entire voice interaction pipeline.

    Features:
    - Multi-agent routing and task delegation
    - Seamless fallback to single-agent mode
    - Preserves all existing audio pipeline functionality
    - Maintains TUI integration and state callbacks
    - Feature flag controlled multi-agent enablement
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[Path] = None,
        text_only: bool = False,
        state_callback: Optional[StateCallback] = None,
    ):
        """
        Initialize the Voice Agent Orchestrator.

        Args:
            config: Configuration object
            config_path: Path to configuration file (if config not provided)
            text_only: Force text-only mode (disables audio pipeline)
            state_callback: Optional callback for pipeline state changes
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
            # Use default config
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

        # Multi-agent mode flag
        self.multi_agent_enabled = (
            self.config.multi_agent.enabled
            if hasattr(self.config, "multi_agent")
            else False
        )

        # Optional pipeline state callback (component, state, message)
        self._state_callback = state_callback

        # Initialize component placeholders
        self.audio_manager: Optional[AudioManager] = None
        self.stt_service: Optional[STTService] = None
        self.tts_service: Optional[TTSService] = None
        self.llm_service: Optional[LLMService] = None
        self.tool_executor: Optional[ToolExecutor] = None

        # Multi-agent components
        self.multi_agent_service: Optional[MultiAgentService] = None
        self.llamaindex_service: Optional[LlamaIndexService] = None

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_running = False
        self._initialized = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        import os

        # Skip logging configuration if debug file logging is already set up
        if os.environ.get("VOICE_AGENT_DEBUG_FILE_LOGGING") == "1":
            # File logging already configured in main.py, don't interfere
            return

        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
        )

    async def initialize(self) -> None:
        """Initialize all components including multi-agent system if enabled."""
        self.logger.info("Initializing Voice Agent Orchestrator...")
        self._emit_state("orchestrator", "initializing", "setting up components")

        # Initialize core pipeline components
        await self._initialize_core_pipeline()

        # Initialize LLM and tools
        await self._initialize_llm_and_tools()

        # Initialize multi-agent system if enabled
        if self.multi_agent_enabled:
            await self._initialize_multi_agent_system()
        else:
            self.logger.info("Multi-agent mode disabled - using single-agent mode")

        self._initialized = True
        self.logger.info("Voice Agent Orchestrator initialized successfully")
        self._emit_state("orchestrator", "ready", "all components initialized")

    async def _initialize_core_pipeline(self) -> None:
        """Initialize core audio pipeline components."""
        if not self.text_only:
            self.logger.info("Initializing audio pipeline components...")

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

            # Initialize TTS service
            self.tts_service = TTSService(
                self.config.tts, self.audio_manager, state_callback=self._state_callback
            )
            await self.tts_service.initialize()

            # Wire barge-in / interruption between audio manager & TTS
            await self._setup_audio_interruption()
        else:
            self.logger.info("Text-only mode: skipping audio pipeline initialization")

    async def _setup_audio_interruption(self) -> None:
        """Setup audio interruption and barge-in functionality."""
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

    async def _initialize_llm_and_tools(self) -> None:
        """Initialize LLM service and tool executor."""
        # Initialize LLM service (for single-agent fallback)
        self.llm_service = LLMService(
            self.config.llm, state_callback=self._state_callback
        )
        await self.llm_service.initialize()

        # Initialize tool executor
        self.tool_executor = ToolExecutor(self.config.tools)
        await self.tool_executor.initialize()

    async def _initialize_multi_agent_system(self) -> None:
        """Initialize multi-agent system components."""
        try:
            self.logger.info("Initializing multi-agent system...")

            # Initialize LlamaIndex service for single-agent fallback
            self.llamaindex_service = LlamaIndexService(
                self.config.llm, state_callback=self._state_callback
            )
            await self.llamaindex_service.initialize()

            # Initialize multi-agent service
            self.multi_agent_service = MultiAgentService(
                config=self.config,
                tool_executor=self.tool_executor,
                llamaindex_service=self.llamaindex_service,
                state_callback=self._state_callback,
            )
            await self.multi_agent_service.initialize()

            self.logger.info("Multi-agent system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize multi-agent system: {e}")
            self.logger.info("Falling back to single-agent mode")
            self.multi_agent_enabled = False

    def _emit_state(
        self, component: str, state: str, message: Optional[str] = None
    ) -> None:
        """Emit state change via callback."""
        if self._state_callback:
            try:
                self._state_callback(component, state, message)
            except Exception:
                self.logger.debug("State callback error", exc_info=True)

    async def start(self) -> None:
        """Start the voice agent orchestrator main loop."""
        required = [self.tool_executor]
        if not self.text_only:
            required.extend([self.audio_manager, self.stt_service, self.tts_service])

        # Always need either LLM service or multi-agent service
        if self.multi_agent_enabled:
            required.append(self.multi_agent_service)
        else:
            required.append(self.llm_service)

        if not all(required):
            await self.initialize()

        self.is_running = True
        self.logger.info("Voice Agent Orchestrator started")
        self._emit_state("orchestrator", "running", "main loop started")

        try:
            await self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Voice Agent stopped by user")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the voice agent orchestrator and cleanup resources."""
        self.is_running = False
        self.logger.info("Stopping Voice Agent Orchestrator...")
        self._emit_state("orchestrator", "stopping", "cleaning up components")

        # Cleanup core pipeline
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

        # Cleanup multi-agent components
        if self.multi_agent_service:
            await self.multi_agent_service.cleanup()
        if self.llamaindex_service:
            await self.llamaindex_service.cleanup()

        self.logger.info("Voice Agent Orchestrator stopped")
        self._emit_state("orchestrator", "stopped", "cleanup complete")

    async def _main_loop(self) -> None:
        """
        Main conversation loop with multi-agent integration.

        In text_only mode this loop idles (input is provided via external calls,
        e.g. TUI adapter invoking process_text). In voice mode it performs the
        full capture → STT → routing → agent processing → TTS pipeline.
        """
        while self.is_running:
            try:
                if self.text_only:
                    # Idle briefly; external components (TUI) drive interactions.
                    await asyncio.sleep(0.25)
                    continue

                # Ensure required components are initialized
                if not self._validate_voice_pipeline():
                    self.logger.warning(
                        "Voice pipeline components missing (reinitializing)..."
                    )
                    await self.initialize()
                    await asyncio.sleep(0.25)
                    continue

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

                # Process through orchestrator (multi-agent or single-agent)
                response = await self._process_user_input(user_input)
                self.logger.info(f"Agent: {response}")

                # TTS
                await self.tts_service.speak(response)

                # Update history
                self._update_history(user_input, response)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    def _validate_voice_pipeline(self) -> bool:
        """Validate that all voice pipeline components are available."""
        if self.text_only:
            return True

        return all(
            [
                self.audio_manager,
                self.stt_service,
                self.tts_service,
                (
                    self.multi_agent_service
                    if self.multi_agent_enabled
                    else self.llm_service
                ),
                self.tool_executor,
            ]
        )

    async def _process_user_input(self, user_input: str) -> str:
        """
        Process user input through the appropriate system (multi-agent or single-agent).

        Args:
            user_input: User's input text

        Returns:
            Agent response text
        """
        if self.multi_agent_enabled and self.multi_agent_service:
            # Use multi-agent system
            try:
                response = await self.multi_agent_service.process_message(user_input)
                return response

            except Exception as e:
                self.logger.error(f"Multi-agent processing failed: {e}")
                self.logger.info("Falling back to single-agent mode for this request")
                # Fallback to single-agent processing
                return await self._process_single_agent(user_input)
        else:
            # Use single-agent mode
            return await self._process_single_agent(user_input)

    async def _process_single_agent(self, user_input: str) -> str:
        """Process user input using single-agent mode."""
        if not self.llm_service:
            raise RuntimeError("LLM service not initialized for single-agent mode")

        response = await self.llm_service.generate_response(
            user_input, self.conversation_history, self.tool_executor
        )
        return response

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

    def set_state_callback(self, callback: Optional[StateCallback]) -> None:
        """Install or replace the pipeline state callback and propagate to components."""
        self._state_callback = callback

        # Propagate to core pipeline components
        if self.audio_manager:
            self.audio_manager.set_state_callback(callback)
        if self.stt_service:
            self.stt_service.set_state_callback(callback)
        if self.tts_service:
            self.tts_service.set_state_callback(callback)
        if self.llm_service:
            self.llm_service.set_state_callback(callback)

        # Propagate to multi-agent components
        if self.multi_agent_service:
            self.multi_agent_service._state_callback = callback
        if self.llamaindex_service:
            self.llamaindex_service.set_state_callback(callback)

    async def process_text(self, text: str) -> str:
        """
        Process text input without audio (for testing/debugging/TUI).

        Args:
            text: Input text to process

        Returns:
            Agent response text
        """
        if not self.tool_executor:
            await self.initialize()
            if not self.tool_executor:
                raise RuntimeError("Tool executor failed to initialize")

        # Process through orchestrator
        response = await self._process_user_input(text)
        self._update_history(text, response)
        return response

    @property
    def is_initialized(self) -> bool:
        """Check if the orchestrator is initialized."""
        return self._initialized

    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator and its components."""
        info = {
            "multi_agent_enabled": self.multi_agent_enabled,
            "text_only": self.text_only,
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "conversation_length": len(self.conversation_history),
            "components": {
                "audio_manager": self.audio_manager is not None,
                "stt": self.stt_service is not None,
                "tts": self.tts_service is not None,
                "llm": self.llm_service is not None,
                "tools": self.tool_executor is not None,
                "conversation": True,  # Always available as part of orchestrator
                "multi_agent_service": self.multi_agent_service is not None,
                "llamaindex_service": self.llamaindex_service is not None,
            },
        }

        # Add multi-agent service info if available
        if self.multi_agent_service:
            info["multi_agent_info"] = self.multi_agent_service.get_service_info()

        return info

    def enable_multi_agent_mode(self) -> None:
        """Enable multi-agent mode (requires reinitialization)."""
        if not self.multi_agent_enabled:
            self.logger.info("Enabling multi-agent mode")
            self.config.multi_agent.enabled = True
            self.multi_agent_enabled = True
            # Reinitialization will be handled on next process_text call

    def disable_multi_agent_mode(self) -> None:
        """Disable multi-agent mode and cleanup multi-agent components."""
        if self.multi_agent_enabled:
            self.logger.info("Disabling multi-agent mode")
            self.config.multi_agent.enabled = False
            self.multi_agent_enabled = False
            # Components will be cleaned up on next reinitialization


# Backward compatibility alias
VoiceAgent = VoiceAgentOrchestrator
