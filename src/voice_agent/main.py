"""
CLI entry point for the voice_agent package.

Resolves configuration relative to the installed package so it works when
run from the project root, an installed environment, or any CWD:
  - `python -m voice_agent.main`
  - `voice-agent` (console script)
  - `python src/main.py` (top-level helper, see duplicated logic there)

Future improvement: unify this with the repository-level src/main.py.
"""

import asyncio
from pathlib import Path

import click

from .core.config import Config
from .core.conversation import VoiceAgent


def _locate_default_config() -> Path:
    """
    Locate the default.yaml shipped with the package.

    Search order:
      1. Package-relative: <package_dir>/config/default.yaml
      2. Source layout (project root): src/voice_agent/config/default.yaml
      3. Fallback: voice_agent/config/default.yaml relative to CWD (legacy)

    Returns:
        Path to existing config file or raises FileNotFoundError.
    """
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "config" / "default.yaml",
        Path("src/voice_agent/config/default.yaml"),
        Path("voice_agent/config/default.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate default.yaml in any known location. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
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
    """Start the Voice Agent (package entrypoint)."""
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
