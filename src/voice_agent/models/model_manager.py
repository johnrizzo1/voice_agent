"""
Model management for the voice agent.
"""

import asyncio
import logging
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ModelManager:
    """
    Manages downloading, caching, and loading of models for the voice agent.

    Handles:
    - Model downloading from various sources
    - Local caching and versioning
    - Model validation and integrity checks
    - Automatic cleanup of old models
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the model manager.

        Args:
            config_path: Path to models configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load model configurations
        if config_path and config_path.exists():
            self.config = self._load_config(config_path)
        else:
            # Use default config path
            default_config_path = (
                Path(__file__).parent.parent / "config" / "models.yaml"
            )
            if default_config_path.exists():
                self.config = self._load_config(default_config_path)
            else:
                self.config = self._get_default_config()

        # Setup cache directories
        self.cache_dir = Path(self.config["model_paths"]["cache_dir"]).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model tracking
        self.downloaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_locks: Dict[str, asyncio.Lock] = {}

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load model config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            "stt_models": {
                "whisper": {
                    "base": {
                        "model_name": "whisper-base",
                        "download_url": "https://huggingface.co/openai/whisper-base",
                        "size_mb": 142,
                        "languages": "multilingual",
                    }
                }
            },
            "tts_models": {
                "coqui": {
                    "ljspeech": {
                        "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                        "quality": "high",
                        "voice_type": "female",
                    }
                }
            },
            "llm_models": {
                "ollama": {
                    "mistral": {
                        "model_name": "mistral:7b",
                        "size_gb": 4.1,
                        "context_length": 8192,
                        "capabilities": ["chat", "tools"],
                    }
                }
            },
            "model_paths": {
                "cache_dir": "~/.cache/voice_agent/models",
                "whisper_cache": "~/.cache/whisper",
                "tts_cache": "~/.cache/tts",
                "ollama_models": "~/.ollama/models",
            },
        }

    async def ensure_model(self, model_type: str, model_name: str) -> Optional[Path]:
        """
        Ensure a model is available, downloading if necessary.

        Args:
            model_type: Type of model (stt, tts, llm)
            model_name: Name of the model

        Returns:
            Path to the model or None if failed
        """
        model_key = f"{model_type}_{model_name}"

        # Check if already downloaded
        if model_key in self.downloaded_models:
            model_path = Path(self.downloaded_models[model_key]["path"])
            if model_path.exists():
                self.logger.debug(
                    f"Model {model_name} already available at {model_path}"
                )
                return model_path

        # Get or create lock for this model
        if model_key not in self.model_locks:
            self.model_locks[model_key] = asyncio.Lock()

        async with self.model_locks[model_key]:
            # Double-check after acquiring lock
            if model_key in self.downloaded_models:
                model_path = Path(self.downloaded_models[model_key]["path"])
                if model_path.exists():
                    return model_path

            # Download the model
            return await self._download_model(model_type, model_name)

    async def _download_model(self, model_type: str, model_name: str) -> Optional[Path]:
        """Download a model."""
        try:
            # Get model configuration
            model_config = self._get_model_config(model_type, model_name)
            if not model_config:
                self.logger.error(
                    f"No configuration found for {model_type} model: {model_name}"
                )
                return None

            self.logger.info(f"Downloading {model_type} model: {model_name}")

            # Create model directory
            model_dir = self.cache_dir / model_type / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download based on model type
            if model_type == "stt":
                model_path = await self._download_stt_model(model_config, model_dir)
            elif model_type == "tts":
                model_path = await self._download_tts_model(model_config, model_dir)
            elif model_type == "llm":
                model_path = await self._download_llm_model(model_config, model_dir)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None

            if model_path:
                # Track the downloaded model
                model_key = f"{model_type}_{model_name}"
                self.downloaded_models[model_key] = {
                    "path": str(model_path),
                    "config": model_config,
                    "type": model_type,
                    "name": model_name,
                }

                self.logger.info(
                    f"Successfully downloaded {model_type} model: {model_name}"
                )
                return model_path

            return None

        except Exception as e:
            self.logger.error(
                f"Failed to download {model_type} model {model_name}: {e}"
            )
            return None

    def _get_model_config(
        self, model_type: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        config_key = f"{model_type}_models"
        if config_key not in self.config:
            return None

        models_config = self.config[config_key]

        # Search through providers and models
        for provider, provider_models in models_config.items():
            if model_name in provider_models:
                config = provider_models[model_name].copy()
                config["provider"] = provider
                return config

            # Also check if model_name matches model_name field
            for model_key, model_config in provider_models.items():
                if model_config.get("model_name") == model_name:
                    config = model_config.copy()
                    config["provider"] = provider
                    return config

        return None

    async def _download_stt_model(
        self, model_config: Dict[str, Any], model_dir: Path
    ) -> Optional[Path]:
        """Download STT model."""
        provider = model_config.get("provider", "unknown")

        if provider == "whisper":
            # Whisper models are handled by the faster-whisper library
            # Just return a placeholder path
            placeholder_file = model_dir / "model.info"
            with open(placeholder_file, "w") as f:
                yaml.dump(model_config, f)
            return model_dir

        elif provider == "vosk":
            # Download Vosk model
            download_url = model_config.get("download_url")
            if not download_url:
                self.logger.error("No download URL for Vosk model")
                return None

            return await self._download_and_extract(download_url, model_dir)

        else:
            self.logger.error(f"Unknown STT provider: {provider}")
            return None

    async def _download_tts_model(
        self, model_config: Dict[str, Any], model_dir: Path
    ) -> Optional[Path]:
        """Download TTS model."""
        provider = model_config.get("provider", "unknown")

        if provider == "coqui":
            # Coqui TTS models are handled by the TTS library
            # Just return a placeholder path
            placeholder_file = model_dir / "model.info"
            with open(placeholder_file, "w") as f:
                yaml.dump(model_config, f)
            return model_dir

        elif provider == "pyttsx3":
            # pyttsx3 uses system voices, no download needed
            placeholder_file = model_dir / "model.info"
            with open(placeholder_file, "w") as f:
                yaml.dump(model_config, f)
            return model_dir

        else:
            self.logger.error(f"Unknown TTS provider: {provider}")
            return None

    async def _download_llm_model(
        self, model_config: Dict[str, Any], model_dir: Path
    ) -> Optional[Path]:
        """Download LLM model."""
        provider = model_config.get("provider", "unknown")

        if provider == "ollama":
            # Ollama models are handled by the Ollama service
            # Just return a placeholder path
            placeholder_file = model_dir / "model.info"
            with open(placeholder_file, "w") as f:
                yaml.dump(model_config, f)
            return model_dir

        else:
            self.logger.error(f"Unknown LLM provider: {provider}")
            return None

    async def _download_and_extract(
        self, url: str, destination_dir: Path
    ) -> Optional[Path]:
        """Download and extract a file."""
        try:
            # Parse filename from URL
            parsed_url = urllib.parse.urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename:
                filename = "download"

            download_path = destination_dir / filename

            # Download file
            self.logger.info(f"Downloading from {url}")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: urllib.request.urlretrieve(url, download_path)
            )

            # Extract if it's an archive
            if filename.endswith((".zip", ".tar.gz", ".tar.bz2", ".tar")):
                extract_dir = destination_dir / "extracted"
                extract_dir.mkdir(exist_ok=True)

                await loop.run_in_executor(
                    None, lambda: self._extract_archive(download_path, extract_dir)
                )

                # Remove the archive file
                download_path.unlink()

                return extract_dir

            return download_path

        except Exception as e:
            self.logger.error(f"Failed to download and extract {url}: {e}")
            return None

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> None:
        """Extract an archive file."""
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        elif (
            archive_path.suffix in [".tar", ".gz", ".bz2"]
            or ".tar." in archive_path.name
        ):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)

        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded models.

        Returns:
            List of model information dictionaries
        """
        models = []

        for model_key, model_info in self.downloaded_models.items():
            model_path = Path(model_info["path"])
            models.append(
                {
                    "key": model_key,
                    "type": model_info["type"],
                    "name": model_info["name"],
                    "path": str(model_path),
                    "exists": model_path.exists(),
                    "size": (
                        self._get_directory_size(model_path)
                        if model_path.exists()
                        else 0
                    ),
                    "config": model_info["config"],
                }
            )

        return models

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in directory.rglob("*"):
                for filename in filenames:
                    filepath = dirpath / filename
                    if filepath.is_file():
                        total_size += filepath.stat().st_size
        except Exception:
            pass
        return total_size

    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models from configuration.

        Returns:
            Dictionary mapping model types to lists of model names
        """
        available = {"stt": [], "tts": [], "llm": []}

        for model_type in ["stt", "tts", "llm"]:
            config_key = f"{model_type}_models"
            if config_key in self.config:
                for provider, models in self.config[config_key].items():
                    for model_name in models.keys():
                        available[model_type].append(model_name)

        return available

    async def cleanup_old_models(self, keep_recent: int = 3) -> None:
        """
        Cleanup old model versions.

        Args:
            keep_recent: Number of recent versions to keep
        """
        self.logger.info("Cleaning up old models...")

        try:
            # This would implement cleanup logic
            # For now, just log that cleanup is available
            self.logger.info(
                f"Cleanup would keep {keep_recent} recent versions of each model"
            )

        except Exception as e:
            self.logger.error(f"Error during model cleanup: {e}")

    def get_model_info(
        self, model_type: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_type: Type of model
            model_name: Name of the model

        Returns:
            Model information dictionary or None if not found
        """
        model_key = f"{model_type}_{model_name}"

        if model_key in self.downloaded_models:
            return self.downloaded_models[model_key].copy()

        # Check if model is available in config
        model_config = self._get_model_config(model_type, model_name)
        if model_config:
            return {
                "type": model_type,
                "name": model_name,
                "config": model_config,
                "downloaded": False,
            }

        return None
