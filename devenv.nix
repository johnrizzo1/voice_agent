{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

# Devenv configuration for the voice_agent Python project.
# Goals:
#  - Fully reproducible Python environment (no requirements.txt necessary)
#  - Provide runtime + TTS/STT dependencies (bark, faster-whisper, etc.)
#  - Provide developer tooling (pytest, black, isort, mypy, flake8)
#  - Expose convenient tasks: run, test, lint, typecheck
#  - Ensure PYTHONPATH points to ./src so `python -m voice_agent.main` works
#  - Keep CUDA-capable torch/torchaudio via nixpkgs (subject to availability)

let
  pkgs-unstable = import inputs.nixpkgs-unstable {
    system = pkgs.stdenv.system;
  };
in
{
  packages = with pkgs; [
    git
    portaudio
    espeak-ng
    alsa-utils
    # Core Python environment with runtime & dev dependencies
    (python312.withPackages (
      ps: with ps; [
        # Runtime
        click
        faster-whisper
        numpy
        ollama
        pyaudio
        psutil
        pydantic
        pyttsx3
        pyyaml
        requests
        rich
        # textual
        torch
        torchaudio
        transformers
        tensorflow

        # Dev / QA
        pytest
        pytest-asyncio
        black
        isort
        flake8
        mypy
        # (Add any additional QA tools here)
      ]
    ))
  ];

  # Python language support (provides python, venv helpers, etc.)
  languages.python.enable = true;
  languages.python.version = "3.12";

  # Use an additional venv layer for a few wheels not available (or easier via pip)
  # These are installed ad‑hoc inside the devenv-managed venv.
  languages.python.venv.enable = true;
  languages.python.venv.quiet = true;
  languages.python.venv.requirements = ''
    webrtcvad>=2.0.10
    vosk>=0.3.45
    bark>=0.1.5
    textual>=5.3.0

    # LlamaIndex Core
    llama-index-core>=0.10.0

    # LlamaIndex Ollama Integration
    llama-index-llms-ollama>=0.2.0
    llama-index-embeddings-ollama>=0.2.0

    # LlamaIndex Essential Components for Multi-Agent Functionality
    llama-index-agent-openai>=0.2.0
    llama-index-multi-modal-llms-openai>=0.1.0
    llama-index-program-openai>=0.1.0

    # Additional LlamaIndex utilities
    llama-index-readers-file>=0.1.0
    llama-index-vector-stores-chroma>=0.1.0

    # Supporting dependencies for LlamaIndex
    nest-asyncio>=1.5.0
  '';

  # Ensure src is importable without installing the package (editable style).
  # config.projectRoot isn't defined in this module context; use the project root path (./).
  env = {
    # Use the project source dir directly so `import voice_agent` works.
    PYTHONPATH = "${./src}";
  };

  # Developer convenience tasks (use: devenv tasks run app:<name>)
  tasks."app:run" = {
    exec = ''python -m voice_agent.main --debug'';
  };
  # tasks."app:test" = {
  #   exec = ''pytest -q'';
  # };
  tasks."app:lint" = {
    exec = ''flake8 src'';
  };
  tasks."app:format" = {
    exec = ''black --check src'';
  };
  tasks."app:fix-format" = {
    exec = ''black src && isort src'';
  };
  tasks."app:typecheck" = {
    exec = ''mypy src'';
  };

  # Optional: automatically show useful info when entering the shell
  enterShell = ''
    # Prepend project src to PYTHONPATH so local edits are preferred.
    if [ -n "$PYTHONPATH" ]; then
      export PYTHONPATH="$PWD/src:$PYTHONPATH"
    else
      export PYTHONPATH="$PWD/src"
    fi

    # echo "Devenv Python environment ready."
    # echo "Using PYTHONPATH=$PYTHONPATH"
    # echo "Run:  python -m voice_agent.main --debug"
    # echo "Or use tasks: devenv tasks run app:run | app:test | app:lint | app:typecheck"
  '';

  claude.code.enable = true;

  # Enable formatters / linters via git hooks
  git-hooks.hooks = {
    rustfmt.enable = true;
    nixfmt-rfc-style.enable = true;
    black.enable = true;
    prettier.enable = true;
    # Optionally add flake8 & mypy git hooks (flakeheaven/mypy wrappers not included by default)
  };

  cachix.enable = true;
  cachix.pull = [
    "pre-commit-hooks"
    "nix-community"
  ];
  cachix.push = "cv-ml-cache";

  dotenv.enable = true;
  difftastic.enable = true;
  delta.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
