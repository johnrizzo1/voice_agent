"""
Configuration management for the voice agent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """Audio configuration settings.

    Added VAD / speech detection tuning:
    - vad_aggressiveness: WebRTC VAD mode (0=least, 3=most aggressive)
    - min_speech_frames: Consecutive (or accumulated) speech frames before confirming speech
    - max_silence_frames: Silence frames after speech to consider utterance ended
    - speech_detection_cooldown: Cooldown (s) after TTS playback before listening resumes
    """

    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 16000
    chunk_size: int = 1024

    # VAD / speech detection parameters
    vad_aggressiveness: int = 1
    min_speech_frames: int = 5
    max_silence_frames: int = 50
    speech_detection_cooldown: float = 1.0


class STTConfig(BaseModel):
    """Speech-to-Text configuration settings."""

    model: str = "whisper-base"
    language: str = "auto"
    streaming: bool = True


class TTSConfig(BaseModel):
    """Text-to-Speech configuration settings.

    Added latency tuning parameters (for Bark and other backends):
    - tts_cooldown_margin: Extra safety margin (seconds) after playback (used for future fine-grained control)
    - post_tts_cooldown: Short cooldown replacing large fixed sleeps (was 2.0s)
    - enable_tts_buffer_double_clear: Optional second input buffer flush

    Bark voice control:
    - bark_voice_preset: Stable speaker identity (maps to Bark history prompt / preset).
      Examples (depending on installed Bark version/presets):
        "v2/en_speaker_1", "v2/en_speaker_6", "v2/en_speaker_9"
      Leave None to allow Bark's default/random behavior.
    """

    engine: str = "coqui"
    voice: str = "default"
    speed: float = 1.0

    # New tuning / latency controls
    tts_cooldown_margin: float = 0.25
    post_tts_cooldown: float = 0.3
    enable_tts_buffer_double_clear: bool = False

    # Bark-specific deterministic voice preset (history prompt). None = default/random.
    bark_voice_preset: Optional[str] = None


class LLMConfig(BaseModel):
    """Language model configuration settings."""

    provider: str = "ollama"
    model: str = "mistral:7b"
    temperature: float = 0.7
    max_tokens: int = 2048


class ToolsConfig(BaseModel):
    """Tools configuration settings."""

    enabled: List[str] = Field(default_factory=list)
    disabled: List[str] = Field(default_factory=list)


class ConversationConfig(BaseModel):
    """Conversation management configuration."""

    max_history: int = 50
    context_window: int = 4096
    interrupt_enabled: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class UIConfig(BaseModel):
    """UI / TUI configuration settings.

    mode:
        - "auto"  : Decide based on CLI flags / environment (default)
        - "tui"   : Force launch of TUI (if dependencies available)
        - "cli"   : Force non-TUI CLI interaction
    force_text_only:
        When True, skip audio / STT / TTS initialization even if those components
        are available (pure text interaction). This is useful for headless servers
        or when focusing on rapid iteration of LLM / tool features.
    refresh_rate:
        Polling / status bar refresh cadence (seconds) in the TUI.
    max_messages:
        Maximum number of chat messages retained in the scrollback panel.
    show_timestamps:
        Display timestamps in chat log.
    color_scheme:
        Named color scheme (future expansion: "default", "light", "high_contrast", etc.)
    enable_animations:
        Enable subtle ACTIVE state animations / blinking indicators (if supported).
    enable_audio:
        Attempt to integrate voice pipeline inside TUI session (future hybrid mode).
    keymap_overrides:
        Dict for remapping / adding key bindings at runtime (e.g. {"f5": "clear_chat"}).
    """

    mode: str = "auto"
    force_text_only: bool = True
    refresh_rate: float = 0.5
    max_messages: int = 500
    show_timestamps: bool = True
    color_scheme: str = "default"
    enable_animations: bool = True
    enable_audio: bool = False
    keymap_overrides: Dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class for the voice agent."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """
        Load configuration from a YAML file, performing lightweight migration
        for missing sections (e.g., 'ui').

        Args:
            config_path: Path to the configuration file

        Returns:
            Config instance
        """
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Migration: ensure 'ui' section exists (added after early versions)
        if "ui" not in raw:
            logging.getLogger(__name__).warning(
                "Config migration: adding missing 'ui' section with defaults"
            )
            raw["ui"] = {}

        return cls(**raw)

    def save(self, config_path: Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path to save the configuration file
        """
        config_data = self.model_dump()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
