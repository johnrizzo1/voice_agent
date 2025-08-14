"""
Repository-level helper entry point for the Voice Agent application.

Delegates to the same logic as voice_agent.main but provided here so developers
can still run `python src/main.py` during local iteration without installing
the package. Config resolution matches the package entry point.
"""

import asyncio
from pathlib import Path

import click

from voice_agent.core.config import Config
from voice_agent.core.conversation import VoiceAgent


def _locate_default_config() -> Path:
    from voice_agent import __file__ as va_init  # type: ignore

    pkg_dir = Path(va_init).resolve().parent
    candidates = [
        pkg_dir / "config" / "default.yaml",
        Path("src/voice_agent/config/default.yaml"),
        Path("voice_agent/config/default.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate default.yaml (searched package & source paths)"
    )


@click.command()
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to configuration file (defaults to package default.yaml)",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Enable debug mode",
)
@click.option(
    "--no-audio",
    is_flag=True,
    help="Disable audio input/output (text-only mode)",
)
def main(config: str | None, debug: bool, no_audio: bool):
    """Start the Voice Agent."""
    if config:
        config_path = Path(config)
        if not config_path.exists():
            click.echo(f"Configuration file not found: {config_path}")
            return
    else:
        try:
            config_path = _locate_default_config()
        except FileNotFoundError as e:
            click.echo(f"❌ {e}")
            return

    try:
        agent_config = Config.load(config_path)

        if debug:
            agent_config.logging.level = "DEBUG"

        if no_audio:
            agent_config.audio.input_device = None
            agent_config.audio.output_device = None

        agent = VoiceAgent(config=agent_config)

        click.echo(f"Using config: {config_path}")
        click.echo("🎤 Voice Agent starting...")
        click.echo("Press Ctrl+C to stop")

        asyncio.run(agent.start())

    except KeyboardInterrupt:
        click.echo("\n👋 Voice Agent stopped")
    except Exception as e:
        click.echo(f"❌ Error starting Voice Agent: {e}")
        if debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
