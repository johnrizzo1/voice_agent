"""
Configuration management for the voice agent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """Audio configuration settings."""

    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 16000
    chunk_size: int = 1024


class STTConfig(BaseModel):
    """Speech-to-Text configuration settings."""

    model: str = "whisper-base"
    language: str = "auto"
    streaming: bool = True


class TTSConfig(BaseModel):
    """Text-to-Speech configuration settings."""

    engine: str = "coqui"
    voice: str = "default"
    speed: float = 1.0


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


class Config(BaseModel):
    """Main configuration class for the voice agent."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Config instance
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

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
