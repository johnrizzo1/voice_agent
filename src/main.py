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
    help="Disable audio input/output (force text-only, even in TUI)",
)
@click.option(
    "--cli",
    is_flag=True,
    help="Force classic CLI mode (disable TUI).",
)
@click.option(
    "--tui",
    is_flag=True,
    help="(Deprecated) Explicitly launch Text UI (now default).",
)
def main(config: str | None, debug: bool, no_audio: bool, cli: bool, tui: bool):
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
        else:
            # Suppress info/debug output unless --debug specified
            agent_config.logging.level = "WARNING"

        if no_audio:
            agent_config.audio.input_device = None
            agent_config.audio.output_device = None

        # Interface mode (default TUI unless --cli provided)
        if tui:
            click.echo("⚠️  '--tui' flag is deprecated (TUI is now default).")
        if tui and cli:
            click.echo("❌ Cannot specify both --tui and --cli.")
            return
        use_tui = not cli
        if use_tui:
            try:
                import importlib

                importlib.import_module("textual")
                importlib.import_module("rich")
            except ImportError as e:
                click.echo(
                    "❌ TUI dependencies not installed. Falling back to CLI. Install with: pip install textual rich"
                )
                click.echo(f"Import error: {e}")
                use_tui = False

        agent = VoiceAgent(config=agent_config)

        click.echo(f"Using config: {config_path}")
        if use_tui:
            click.echo("🖥️  Launching Voice Agent TUI...")
            click.echo(" - Type or speak. Use 'Privacy Mode' voice command to pause.")
            click.echo(" - Ctrl+Q to quit.")
            asyncio.run(_run_tui(agent))
        else:
            click.echo("🎤 Voice Agent starting (CLI mode)...")
            click.echo("Press Ctrl+C to stop")
            asyncio.run(agent.start())

    except KeyboardInterrupt:
        click.echo("\n👋 Voice Agent stopped")
    except Exception as e:
        click.echo(f"❌ Error starting Voice Agent: {e}")
        if debug:
            import traceback

            traceback.print_exc()


async def _run_tui(agent: VoiceAgent) -> None:
    """Initialize (if needed) and launch the TUI in text-only mode."""
    if not agent.llm_service:
        await agent.initialize()
    from voice_agent.ui.tui_app import run_tui  # type: ignore

    await run_tui(agent)


if __name__ == "__main__":
    main()
