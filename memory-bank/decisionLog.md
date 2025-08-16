# Decision Log

This file records architectural and implementation decisions using a list format.

## Decision

2025-08-14 18:20:18 - **PyTorch 2.6 Compatibility Fix: Monkey Patch Approach**

## Rationale

The Bark TTS library uses `torch.load()` internally, and PyTorch 2.6 changed the default `weights_only` parameter from `False` to `True`. This broke Bark model loading because the models contain numpy objects not on the default allowlist.

Alternative approaches considered:

1. Using `torch.serialization.safe_globals()` context manager - Failed because it only affects the current thread context, not the library's internal calls
2. Patching the bark library directly - Too invasive and would require maintaining a fork
3. Downgrading PyTorch - Would conflict with other dependencies and reduce security benefits
4. Monkey patching torch.load temporarily - **CHOSEN** as it's minimally invasive and preserves security for other uses

## Implementation Details

- Modified `_initialize_bark()` method in `src/voice_agent/core/tts_service.py`
- Created `safe_preload()` function that temporarily replaces `torch.load` with a version that forces `weights_only=False`
- Used try/finally block to ensure original `torch.load` is always restored
- Added proper error handling and type checking for cases where torch might be None
- Patch is applied only during Bark model loading, maintaining security for other torch.load calls

---

## Decision

2025-08-14 18:20:18 - **Development Environment: devenv/nix over requirements.txt**

## Rationale

The project uses devenv/nix for reproducible development environments instead of traditional Python requirements.txt. This provides:

- Fully reproducible builds across different systems
- Integrated system dependencies (portaudio, espeak-ng, etc.)
- CUDA support configuration
- Development tools integration (linters, formatters)

## Implementation Details

- Configured `devenv.nix` with all runtime and development dependencies
- Used both nixpkgs Python packages and venv requirements for packages not available in nixpkgs
- Set up proper PYTHONPATH configuration to use local source directory
- Added convenient process definitions for common development tasks (run, test, lint)

---

## Decision

2025-08-14 22:24:06 - **Multi-Backend TTS Architecture with Automatic Fallback**

## Rationale

Rather than relying on a single TTS engine, implemented a multi-backend architecture to ensure reliability and quality options:

1. **Bark TTS** (Primary): Neural network voices, most realistic quality
2. **Coqui TTS** (Secondary): High-quality neural voices, good fallback
3. **eSpeak-NG** (Tertiary): Lightweight, fast, synthetic but clear
4. **pyttsx3** (Fallback): System TTS, always available, cross-platform

This provides graceful degradation and allows users to choose quality vs. speed trade-offs.

## Implementation Details

- TTS service automatically detects available engines at initialization
- Engine priority: Bark → Coqui → eSpeak-NG → pyttsx3
- Configuration allows explicit engine selection via `engine: "auto"` or specific name
- All engines work completely offline after initial model downloads
- Error handling ensures fallback to next available engine if primary fails

---

## Decision

2025-08-14 22:24:06 - **STT Engine Selection: faster-whisper as Primary**

## Rationale

Selected faster-whisper over alternatives for primary STT:

**faster-whisper** (Primary):

- Optimized Whisper implementation with significant speed improvements
- Same accuracy as original Whisper but with better performance
- Good model size options (base, small, medium, large-v2)
- Excellent offline capabilities

**Vosk** (Alternative):

- Lightweight, good for streaming applications
- Lower accuracy than Whisper but faster startup
- Better for resource-constrained environments

## Implementation Details

- faster-whisper configured with automatic model downloads
- Model size configurable: base (39MB), small (244MB), medium (769MB), large-v2 (1550MB)
- CPU/GPU acceleration support
- Streaming capability for real-time transcription

---

## Decision

2025-08-14 22:24:06 - **LLM Integration: Ollama as Primary Provider**

## Rationale

Chose Ollama over direct model integration for several reasons:

**Advantages of Ollama**:

- Simplified model management (download, switching, updates)
- Optimized inference with automatic hardware acceleration
- RESTful API for clean separation of concerns
- Wide model support (Mistral, Llama, CodeLlama, etc.)
- Better resource management and concurrent request handling

**Alternative Considered**: llama-cpp-python for direct integration

- More complex model management
- Requires more manual optimization
- Better for embedded scenarios but more complex for development

## Implementation Details

- Ollama service runs independently on localhost:11434
- LLM service communicates via HTTP API
- Model selection configurable (mistral:7b default for balance of speed/quality)
- Function calling support for tool integration
- Context management for conversation memory

---

## Decision

2025-08-14 22:24:06 - **Tool Framework: Plugin-Based Architecture with Automatic Discovery**

## Rationale

Designed extensible tool framework to support easy addition of new capabilities:

**Key Design Principles**:

- Plugin-based: Tools can be added without modifying core code
- Automatic Discovery: Tools register themselves via decorators or class inheritance
- Schema Generation: Tool parameters automatically documented for LLM function calling
- Sandboxed Execution: Tools run with proper error handling and isolation
- Type Safety: Pydantic models for parameter validation

## Implementation Details

- Base `Tool` class with standard interface (`execute` method, parameter schemas)
- `@tool` decorator for simple function-based tools
- Tool registry for automatic discovery and registration
- Built-in tools: calculator, weather, file operations, system info
- JSON schema generation for LLM function calling integration
- Error handling prevents tool failures from crashing agent

---

## Decision

2025-08-14 22:24:06 - **Configuration Management: YAML-Based with Environment Override**

## Rationale

Implemented hierarchical configuration system for flexibility:

**Configuration Sources** (priority order):

1. Environment variables (highest priority)
2. Command-line arguments
3. Configuration files (YAML)
4. Default values (lowest priority)

**Benefits**:

- Easy deployment configuration changes
- Development vs. production environment support
- User customization without code changes
- Validation and type checking via Pydantic

## Implementation Details

- Primary config: `voice_agent/config/default.yaml`
- Pydantic models for configuration validation
- Environment variable support (e.g., `VOICE_AGENT_DEBUG=1`)
- Runtime configuration updates for testing/debugging
- Audio device configuration with automatic detection fallback

---

## Decision

2025-08-14 22:24:06 - **Testing Strategy: Multi-Level Testing Approach**

## Rationale

Implemented comprehensive testing strategy addressing different aspects:

**Testing Levels**:

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **System Tests**: End-to-end conversation flows
4. **Performance Tests**: Latency and throughput benchmarks
5. **Manual Testing**: Interactive testing tools

**Benefits**:

- Early detection of component failures
- Regression prevention during development
- Performance monitoring and optimization guidance
- User acceptance testing capabilities

## Implementation Details

- Integration testing guide with step-by-step component verification
- Automated test scripts for common scenarios
- Performance benchmarking tools for STT/TTS latency measurement
- Interactive testing modes (`--debug`, `--no-audio`)
- Component isolation testing for troubleshooting
  2025-08-14 19:18:50 - **Audio Format Fix: Bark Float32 → PCM16 Conversion Before Playback**

Rationale:
Bark generated IEEE float32 WAV data (format code 3). Python's wave module and downstream playback/load path expected PCM (format code 1), producing log: "Error loading audio file: unknown format: 3". Converting to int16 ensures broad compatibility and eliminates runtime error.

Implementation Details:

- Updated [`_speak_bark()`](src/voice_agent/core/tts_service.py:344) to:
  - Clip float32 array to [-1.0, 1.0]
  - Scale to int16 (value \* 32767) and cast
  - Write via scipy wavfile as PCM16
- Added explanatory comments and guarded conversion block
- Verified removal of the error after rerun

Implications:

- Playback reliability improved across environments lacking float WAV support
- Enables future post-processing (e.g., normalization) on consistent PCM16 data
- No change to synthesis latency (conversion negligible versus model generation)

2025-08-15 00:20:30 - **UI Configuration Layer & Text-Only Mode Integration**

Rationale:
Introduced dedicated UIConfig (ui:) section to centralize TUI / interface concerns without polluting core config models. Adds force_text_only flag (default true) enabling pure text interaction (skips audio/STT/TTS) for TUI sessions or headless environments, plus presentation options (refresh_rate, max_messages, color_scheme, animations, keymap overrides) to support forthcoming TUI enhancements (status animations, panels, search, settings overlay).

Implementation Details:

- Added UIConfig pydantic model to [`config.py`](src/voice_agent/core/config.py:101) with fields: mode, force_text_only, refresh_rate, max_messages, show_timestamps, color_scheme, enable_animations, enable_audio, keymap_overrides.
- Extended main [`Config`](src/voice_agent/core/config.py:101) with new ui field and updated [`default.yaml`](src/voice_agent/config/default.yaml:57) to include ui section.
- Updated VoiceAgent initialization to respect ui.force_text_only, gating audio/STT/TTS setup.
- Adjusted main loop to idle in text_only mode allowing external drivers (TUI) to inject input via `process_text()`.
- Updated `run_tui` to pass pipeline_status for future dynamic state updates (ACTIVE/READY transitions).

Implications:

- Decouples presentation-layer tuning from pipeline logic.
- Enables incremental rollout of TUI features (help/settings/debug panels, animations) without additional breaking config changes.
- Simplifies headless CI / development workflows (no ALSA / device warnings).
- Establishes foundation for future hybrid (enable_audio=true) voice+text TUI mode.

Risks / Follow-up:

- Need to propagate timestamp / color preferences into ChatLog rendering (show_timestamps toggle).
- Add validation / migration logic if ui section absent in older user configs.
- Implement key binding override application in TUI startup phase.

## Decision

2025-08-15 01:41:15 - **TUI Audio Toggle Cleanup & Documentation Clarification**

## Rationale

Multiple duplicate initializations of the `_audio_deactivated` flag in [`AgentAdapter.__init__`](src/voice_agent/ui/tui_app.py:1073) accumulated due to partially applied prior diffs, creating noise and potential future confusion about intended state semantics. The `action_push_to_talk` docstring still described outdated modifier-based behaviors not yet implemented, risking misunderstanding of current functionality.

## Implementation Details

- Consolidated 8 duplicate lines initializing `self._audio_deactivated = False` into a single clear line with an explanatory comment in [`AgentAdapter.__init__`](src/voice_agent/ui/tui_app.py:1078).
- Replaced legacy multi-mode docstring on [`VoiceAgentTUI.action_push_to_talk()`](src/voice_agent/ui/tui_app.py:798) with clarified current behavior:
  - First F5: lazy initialize pipeline (audio_manager, STT, TTS) + immediate one-shot capture
  - Second F5: full teardown (deactivation) of audio components
- Removed references to Alt/Shift modifier behaviors (not yet implemented) and future-mode speculation, moving them to a concise "Future enhancements" note.

## Implications

- Reduces cognitive overhead when extending audio toggle logic (e.g., future continuous/hold-to-talk modes).
- Eliminates risk that tooling / static analyzers misinterpret repeated assignments as merge artifacts.
- Aligns in-app help & actual runtime behavior for F5 toggle, improving user discoverability.
- Establishes a clean foundation for upcoming enhancements (continuous capture, modifier-based mode switching).

## Follow-up

- Implement continuous / cancel modes (Shift / hold variants) and update docstring accordingly.
- Persist last audio activation state across sessions (ui.enable_audio) to config (optional).
- Add tests: activate → capture → deactivate sequence; re-activation after teardown.

## Decision

2025-08-15 01:47:20 - **Quit Keybinding Conflict Resolution (Remove Ctrl+C Quit, Retain Ctrl+Q)**

## Rationale

`Ctrl+C` is a ubiquitous terminal copy shortcut (especially in some terminal multiplexers / GUI terminal copy modes) and its interception to quit the TUI prevented users from copying chat/log output for external use. This degraded usability and conflicted with standard expectations. Retaining `Ctrl+Q` as the explicit quit key preserves an accessible keyboard exit while freeing `Ctrl+C` for copy or future SIGINT passthrough behavior.

## Implementation Details

- Removed `("ctrl+c", "quit_app", "Quit")` entry from [`VoiceAgentTUI.BINDINGS`](src/voice_agent/ui/tui_app.py:492).
- Updated input-level handler in [`InputPanel.on_key`](src/voice_agent/ui/tui_app.py:446) to ignore `ctrl+c` (previously triggered quit).
- Updated global key handler in [`VoiceAgentTUI.on_key`](src/voice_agent/ui/tui_app.py:891) to only quit on `ctrl+q`.
- Help text updated to show `Ctrl+Q: Quit` (removed `Ctrl+C / Ctrl+Q: Quit`) at [`_help_text()`](src/voice_agent/ui/tui_app.py:688).
- No change to `action_quit_app`; still used by remaining binding.

## Implications

- Users can now copy text safely without terminating the session.
- Keeps an intentional, less collision‑prone quit accelerator (`Ctrl+Q`).
- Leaves room for future optional SIGINT handling (e.g., abort in‑progress voice capture) mapped to `Ctrl+C` if desired.

## Follow-up (Optional)

- Consider adding `/quit` command as alternative exit path.
- Potential future mapping: `Ctrl+C` to cancel current capture / generation if a cancellable task API is added.
- Document behavior in README TUI usage section.

## Decision

2025-08-15 01:54:55 - **Remove Esc Global Quit Fallback (Ctrl+Q Sole Quit Mechanism)**

## Rationale

After migrating quit behavior to Ctrl+Q (and freeing Ctrl+C for copy), Esc still triggered an application quit when focus was outside the input and no overlays were visible. This created accidental exits during navigation (e.g., dismissing a panel or intending to clear a transient focus). Standard terminal/UI expectations reserve Esc for context clearing or dismissal—not full application termination.

## Implementation Details

- Updated global key handler in [`VoiceAgentTUI.on_key`](src/voice_agent/ui/tui_app.py:885) removing conditional block that quit on bare `escape`.
- Revised docstring to reflect simplified behavior: only Ctrl+Q quits; Esc no longer quits.
- Retained Esc handling exclusively inside [`InputPanel.on_key`](src/voice_agent/ui/tui_app.py:446) to clear the current input line.
- Help text already consistent (Esc listed only for clearing input line).

## Implications

- Eliminates accidental termination via exploratory Esc presses.
- Aligns with conventional TUI/terminal usability (Esc = cancel/clear, not exit).
- Simplifies mental model: single explicit quit chord (Ctrl+Q).
- Frees Esc for future enhancements (e.g., dismiss active overlay, exit search mode).

## Follow-up (Optional)

- Extend Esc to: close top-most visible overlay (help/settings/debug/search/tool/metrics) if any; otherwise no-op.
- Add `/quit` slash command for accessibility / scripting.
- Consider `:q` ex-style command if command palette is implemented.

## Decision

2025-08-14 22:13:55 - **Dictation Mode Introduction & Keybinding Reshuffle**

## Rationale

Need for long-form user speech input without holding a key and distinct from one-shot push-to-talk to enable natural multi-sentence turns before LLM processing. Existing F5 push-to-talk model insufficient for extended narration. Also required freeing/realigning function keys to accommodate new mode and future streaming/interrupt controls.

## Implementation Details

- Added continuous dictation mode toggled via F6 (`VoiceAgentTUI.action_toggle_dictation()`).
- Introduced internal state fields: `_dictation_mode_active`, `_dictation_task`, `_dictation_segments`, `_dictation_preview_msg`.
- Implemented `_dictation_loop()` performing sequential microphone capture → STT transcription → incremental aggregation into a preview ChatMessage ending with cursor glyph.
- On stop (second F6) preview message finalized into a single user message and normal LLM request flow triggered.
- Suppressed push-to-talk (F5) while dictation active: push attempts add system notice instead of capturing.
- Keymap shifts: F6 (dictation), F7 (timestamps), F8 (color cycle), F9 (animations), F10 (export chat), F11 (export logs), F12 (metrics), Ctrl+Shift+L (clear logs).
- Added `action_clear_logs` and local `_record_metric()` helper in TUI to record per-segment STT latencies (avoids prior AttributeError referencing adapter-only metric method).

## Implications

- Establishes foundation for future streaming LLM output + interruption controls (planned).
- Normalizes long-form turn creation, improving conversational coherence.
- Distinguishes transient (push) vs aggregated (dictation) speech entry paths.
- Keybinding clarity improved; reduces future collision risk for added features.

## Follow-up

- Add dictation cancel (discard) shortcut (e.g. Esc during dictation).
- Optional inactivity timeout & max duration safeguards.
- Streaming partial LLM response surface + interrupt key (Esc / Ctrl+Break).
- Tests: segment accumulation, suppression of push-to-talk, metrics recording validation.
- Persist dictation-enable state if desired across sessions (config.ui).

---

## Decision

2025-08-14 22:13:55 - **Local Metrics Recording Helper in TUI**

## Rationale

Dictation loop required lightweight latency measurements (STT segment times) without routing through AgentAdapter (which originally owned metric recording). Missing method caused runtime: `'VoiceAgentTUI' object has no attribute '_record_metric'`.

## Implementation Details

- Added `_record_metric(phase, ms)` to `VoiceAgentTUI` appending entries to `_metrics` deque and refreshing metrics panel.
- Invoked from dictation loop after each STT segment transcription.

## Implications

- Enables immediate performance visibility for dictation use-case.
- Decouples UI metric needs from adapter internals, reducing cross-class coupling.

## Follow-up

- Extend metrics to include word rate, segment count, end-to-end dictation duration.
- Aggregate rolling statistics (avg STT latency last N segments).
- Unify adapter & TUI metric collection behind shared utility (future refactor).

---

2025-08-15 02:41:57 - **TUI Chat Scrolling Refactor (Manual Virtualization → Native ScrollView)**

Rationale:
Manual virtualized slicing in [`ChatLog.render()`](src/voice_agent/ui/tui_app.py:322) prevented native Textual 4.0.0 scrollbar display and hardware/mouse / PageUp/PageDown scrolling on user system. Autoscroll (tail follow) also failed because `_line_offset` logic fought with ScrollView mechanics and no `scroll_end()` was invoked post message append. This produced user-visible issues: no scrollbar, no paging effect, no automatic tail follow.

Implementation Details:

- Removed manual viewport height computations, `_line_offset` pagination math, and footer injection in [`ChatLog.render()`](src/voice_agent/ui/tui_app.py:322).
- Simplified render to emit full formatted transcript letting ScrollView manage virtualization internally.
- Adjusted [`ChatLog.add_message()`](src/voice_agent/ui/tui_app.py:281) to call `scroll_end(animate=False)` after refresh when user has not manually scrolled (auto-follow behavior).
- Rewrote scroll actions (`action_scroll_chat_up/down/top/bottom`) (around [`VoiceAgentTUI.action_scroll_chat_up()`](src/voice_agent/ui/tui_app.py:1082)) to use native `scroll_relative`, `scroll_to`, and `scroll_end` instead of mutating `_line_offset`.
- Removed diagnostic system messages previously appended for scroll offset debugging.
- Retained `_user_scrolled` flag only as a gate to suppress auto-follow after manual navigation; `_line_offset` no longer drives rendering logic (legacy references eliminated in actions).
- Preserved highlight, color scheme, timestamp toggle, and search match emphasis logic.

Implications:

- Restores native scrollbar visibility (Textual-managed).
- Enables mouse wheel and keyboard paging to function with built-in mechanics.
- Reduces custom layout error surface and future maintenance burden.
- Eliminates footer page indicator; future enhancement could reintroduce a lightweight status line via separate widget if needed.
- Potential increased memory usage for very large histories (full render) mitigated by existing `max_messages` + pruning strategy.

Follow-up:

- Validate on target environment that scrollbar now appears and PageUp/PageDown navigate.
- Consider adding optional mini status line widget for position (e.g., "120/450").
- Remove now-unused `_line_offset` attribute entirely in a later cleanup pass (left transiently for backward compatibility but no longer functional).
- Add tests: append > viewport messages, assert tail follow until manual scroll; after manual scroll ensure additional messages do not shift view; after bottom action tail follow resumes.

---

## Decision

2025-08-15 03:14:45 - **Voice Command Interpretation Layer for Dictation Control**

## Rationale

Need hands-free control of long-form speech input (dictation) without relying on keyboard function keys (F5/F6) so the user can verbally:

- Start dictation
- End / finalize dictation
- Pause (currently treated same as end/finalize)
- Cancel / discard dictation

Key considerations:

1. Low Latency / Offline: A lightweight local pattern match avoids LLM round-trip latency and preserves privacy (no cloud).
2. Safety / False Positives: Limited to a constrained synonym set; matching requires full phrase or phrase suffix to reduce accidental triggers during normal conversation.
3. Incremental Evolution: Simple rule-based layer now; can later be replaced or augmented with intent classification / small local model without changing higher-level TUI or pipeline code.
4. Non-Intrusive: Implemented entirely in UI layer (TUI app + adapter) leaving core VoiceAgent pipeline untouched.

## Implementation Details

- Added static command detector [`VoiceAgentTUI.detect_voice_command()`](src/voice_agent/ui/tui_app.py:1381) returning normalized command keys:
  - start_dictation
  - end_dictation
  - pause_dictation (alias of end for now)
  - cancel_dictation
- Integrated interception points:
  - Push-to-talk one‑shot path inside [`AgentAdapter._do_capture()`](src/voice_agent/ui/tui_app.py:1793) before forwarding transcription to LLM.
  - Continuous dictation loop modifications in [`VoiceAgentTUI._dictation_loop()`](src/voice_agent/ui/tui_app.py:1549) to recognize in-session control phrases.
- Added executor method [`VoiceAgentTUI.handle_voice_command()`](src/voice_agent/ui/tui_app.py:1430) performing state transitions:
  - start_dictation → initializes dictation mode if not already active
  - end_dictation / pause_dictation → finalizes + sends aggregated text
  - cancel_dictation → stops without sending (system notice)
- Ensured settings panel refresh reflects new dictation state after voice-driven transitions.
- Guarded against redundant "start dictation" while already active.
- Command detection strips terminal punctuation and matches either exact string or suffix (norm.endswith()) to tolerate leading filler words.

## Implications

- Enables fully voice-driven long-form input cycles (hands off keyboard once audio pipeline active).
- Establishes an abstraction seam for future richer intent handling (e.g., "pause recording for a moment", "discard that").
- Minimal performance overhead (string operations only) and no additional dependencies.
- Slight risk of inadvertent triggers if user naturally ends a sentence with an identical short control phrase; mitigated by requiring phrase boundary equality/suffix rather than substring.

## Follow-up

1. Distinguish pause vs finalize: Implement paused state preserving accumulated segments without sending (resume command).
2. Add configurable hotword prefix (e.g., "computer, start dictation") to further reduce accidental triggers.
3. Add telemetry / counters for command usage & false positive analysis (optional metrics panel enrichment).
4. Write unit tests for detector: phrase variants, punctuation handling, negative cases.
5. Internationalization: expose command phrase lists via configuration for localization.

---

## Decision

2025-08-15 04:08:30 - **Always-On Continuous Listening & Privacy Mode Voice Commands**

## Rationale

Previously, STT/TTS pipeline activation required explicit key presses (F5 push-to-talk / F6 dictation). For a more natural, hands‑free experience the agent should begin listening automatically at TUI startup and remain passively available. However, continuous capture raises privacy concerns; users need an immediate, low‑latency, fully local method to suspend all audio processing without touching the keyboard. Introducing a simple “Privacy Mode” voice toggle provides an intuitive safeguard.

## Implementation Details

- Added continuous background listener task started during [`VoiceAgentTUI.on_mount()`](src/voice_agent/ui/tui_app.py:803) after lazy pipeline activation.
- Introduced internal flags / tasks:
  - `_privacy_mode: bool` – suppresses microphone processing (no STT, no dictation, no LLM, no TTS) while True.
  - `_continuous_listen_task: asyncio.Task` – passive always-on loop handling VAD → STT → command detection → conversation.
- Extended voice command detector [`detect_voice_command()`](src/voice_agent/ui/tui_app.py:1406) with:
  - `privacy_on`: (“privacy mode”, “privacy mode on”, “stop listening”)
  - `privacy_off`: (“privacy mode off”, “resume listening”)
- Added handling branches inside [`handle_voice_command()`](src/voice_agent/ui/tui_app.py:1478) to toggle privacy state; enabling privacy cancels any active dictation.
- Implemented `_continuous_listen_loop()` (appended near [`_dictation_loop()`](src/voice_agent/ui/tui_app.py:1736)) performing:
  1. Skip if `_privacy_mode` or dictation active.
  2. `audio_manager.listen()` (VAD gated) → STT transcription.
  3. Voice command interception (dictation / privacy).
  4. Normal utterance → LLM generation → optional TTS playback → history update.
- Updated help & settings UI text via `_help_text()` and `_settings_text()` to surface privacy commands and current mode status.
- Dictation loop updated to recognize privacy commands mid-session (auto-cancels if “Privacy Mode” spoken).
- Push‑to‑talk capture now aborts gracefully when privacy mode is active (system notice only).

## Implications

- User gains immediate hands‑free conversational flow without manual activation friction.
- Clear, local, deterministic privacy control (no cloud calls, simple phrase set) reduces risk of unintended transcription.
- Slight increase in idle CPU usage due to passive loop & periodic VAD guarded listens, but mitigated by existing VAD gating & early returns.
- Added command vocabulary marginally increases false positive surface; mitigated by full-phrase / suffix matching strategy.
- Architectural seam established for future wake‑word or hotword gating (could precede continuous loop).

## Follow-up

1. Optional configurable hotword (“computer,”) prefix to reduce accidental privacy toggles in normal speech.
2. Visual indicator (e.g., STATUS BAR badge or color shift) when Privacy Mode is ON.
3. Metrics / counters for time spent in privacy vs active listening.
4. Config exposure (ui.privacy_startup: on/off) to allow default privacy-on launch.
5. Unit tests: command detection (privacy_on/off), loop suppression during privacy, dictation cancellation on privacy activation.
6. Potential low-power / duty-cycle listening (adaptive sleep intervals) when in long idle periods.

---

## Decision

2025-08-15 18:13:55 - **Multi-Agent Framework Implementation with LlamaIndex Integration**

## Rationale

Implemented a comprehensive multi-agent framework built on LlamaIndex to enable specialized agent handling of different query types while maintaining full backward compatibility with the existing voice agent system. This addresses the need for more sophisticated task routing and agent specialization without disrupting current functionality.

## Implementation Details

**Core Components Implemented:**

1. **Message System** (`src/voice_agent/core/multi_agent/message.py`):
   - AgentMessage and AgentResponse data classes with full metadata support
   - MessageRouter for correlation tracking and conversation management
   - Comprehensive error handling and timeout management
   - Message processing history and routing metadata

2. **Base Agent Framework** (`src/voice_agent/core/multi_agent/agent_base.py`):
   - AgentBase abstract class with LlamaIndex ReAct agent integration
   - GeneralAgent and ToolSpecialistAgent concrete implementations
   - Agent capability system with dynamic tool assignment
   - Performance metrics and conversation history management
   - Async initialization with proper resource management

3. **Hybrid Routing System** (`src/voice_agent/core/multi_agent/router.py`):
   - Rule-based routing for explicit pattern matching
   - Embedding-based semantic routing using Ollama embeddings
   - LLM fallback routing for complex decisions (extensible)
   - Load balancing among capable agents
   - Configurable routing rules and confidence thresholds

4. **Shared Context Manager** (`src/voice_agent/core/multi_agent/context.py`):
   - Conversation slicing per agent with relevance scoring
   - Context sharing between agents during handoffs
   - Sliding window context management with token limits
   - RAG integration hooks for future knowledge base access

5. **Tool Adapter** (`src/voice_agent/core/multi_agent/tool_adapter.py`):
   - Seamless bridge between existing ToolExecutor and LlamaIndex FunctionTool
   - Automatic tool conversion with parameter schema mapping
   - Agent-specific tool filtering based on capabilities
   - Custom tool creation support for specialized functions

6. **Configuration Extensions**:
   - MultiAgentConfig model with comprehensive agent definitions
   - YAML configuration schema with routing rules
   - Feature flag support (multi_agent.enabled) for gradual rollout
   - Backward compatibility preservation with existing configs

7. **Integration Service** (`src/voice_agent/core/multi_agent_service.py`):
   - Main orchestration service with automatic fallback to single-agent mode
   - Agent handoff handling with context preservation
   - Performance tracking and routing statistics
   - Complete backward compatibility when multi-agent disabled

## Key Architectural Benefits

- **Backward Compatibility**: Existing voice agent functionality unchanged when multi-agent disabled
- **Gradual Adoption**: Feature flag allows incremental rollout and testing
- **Extensible Architecture**: Easy addition of new agent types and routing strategies
- **Performance Optimization**: Load balancing and agent specialization improve response quality
- **Context Preservation**: Conversation continuity maintained across agent handoffs
- **Tool Integration**: Existing tools work seamlessly with new agent system

## Configuration Structure

```yaml
multi_agent:
  enabled: false # Feature flag
  default_agent: "general_agent"
  routing_strategy: "hybrid"
  agents:
    general_agent:
      type: "GeneralAgent"
      capabilities: ["general_chat", "tool_execution"]
      tools: ["calculator", "weather", "file_ops"]
    tool_specialist:
      type: "ToolSpecialistAgent"
      capabilities: ["tool_execution", "file_operations", "calculations"]
      tools: ["calculator", "file_ops", "weather", "web_search"]
  routing_rules:
    - name: "calculation_requests"
      target_agent: "tool_specialist"
      patterns: ["calculate", "compute", "math"]
```

## Testing and Validation

- Comprehensive integration test suite created (`test_multi_agent_integration.py`)
- All 6 test categories passed successfully in devenv environment
- Verified backward compatibility with existing system
- Confirmed graceful fallback when dependencies unavailable
- Validated configuration loading and tool integration

## Implementation Patterns Used

- **Service Architecture Pattern**: Independent, composable services with clear interfaces
- **Adapter Pattern**: Tool adaptation between existing and LlamaIndex formats
- **Strategy Pattern**: Pluggable routing strategies (rules, embeddings, LLM)
- **Observer Pattern**: State callbacks for monitoring and debugging
- **Factory Pattern**: Agent creation based on configuration
- **Context Manager Pattern**: Automatic resource cleanup and lifecycle management

## Follow-up Opportunities

1. **LLM-based Routing**: Complete implementation of LLM fallback routing
2. **Advanced Agent Types**: Code analysis, web search, and planning specialists
3. **RAG Integration**: Knowledge base integration via vector stores
4. **Performance Optimization**: Caching and parallel agent processing
5. **Monitoring Dashboard**: Real-time agent performance and routing visualization
6. **Agent Learning**: Adaptive routing based on success metrics

## Implications

- Enables sophisticated multi-turn conversations with agent specialization
- Provides foundation for future AI assistant capabilities (planning, research, analysis)
- Maintains development velocity through backward compatibility
- Establishes patterns for complex AI system architecture in voice agents
- Sets stage for advanced features like agent collaboration and workflow orchestration

---

## Decision

2025-08-15 18:39:00 - **InformationAgent Implementation for Multi-Agent Voice System**

## Rationale

Implemented a specialized InformationAgent to enhance the multi-agent voice system with dedicated information retrieval capabilities. This addresses the need for:

1. **Specialized Information Processing**: Weather, web search, and news queries require domain-specific expertise
2. **Enhanced Response Formatting**: Information responses need context-aware formatting and source attribution
3. **Intelligent Query Routing**: Information queries should be routed to agents with specialized knowledge
4. **Future Extensibility**: Framework for adding news, RSS, and other information sources

## Implementation Details

**Core InformationAgent Features**:

- Extends [`AgentBase`](src/voice_agent/core/multi_agent/agent_base.py:525) with information-focused capabilities
- Specialized in [`WEATHER_INFO`](src/voice_agent/core/multi_agent/agent_base.py:47), [`WEB_SEARCH`](src/voice_agent/core/multi_agent/agent_base.py:44), and [`NEWS_INFO`](src/voice_agent/core/multi_agent/agent_base.py:48) capabilities
- Enhanced system prompts optimized for information retrieval tasks
- Smart handoff logic to route non-information queries to appropriate agents
- Response formatting with context-aware tips and source attribution

**Tool Integration**:

- Uses existing [`WeatherTool`](src/voice_agent/tools/builtin/weather.py:23) for weather queries
- Uses existing [`WebSearchTool`](src/voice_agent/tools/builtin/web_search.py:26) for web searches
- Added placeholder [`NewsTool`](src/voice_agent/tools/builtin/news.py:39) for future RSS/news API integration

**Configuration and Routing**:

- Added InformationAgent configuration to [`MultiAgentConfig`](src/voice_agent/core/config.py:153)
- Created high-priority routing rules for information queries in [`default.yaml`](src/voice_agent/config/default.yaml:120)
- Updated multi-agent service to support InformationAgent creation

**Agent Module Structure**:

- Created [`src/voice_agent/agents/__init__.py`](src/voice_agent/agents/__init__.py:1) for agent exports
- Updated [`multi_agent/__init__.py`](src/voice_agent/core/multi_agent/__init__.py:17) to include concrete agent classes

## Key Architectural Benefits

- **Domain Expertise**: Specialized agent for information-related queries improves response quality
- **Response Enhancement**: Information-specific formatting and context-aware tips
- **Smart Routing**: High-priority routing rules ensure information queries reach the right agent
- **Tool Specialization**: Focused tool assignment (weather, web_search, news) for optimal performance
- **Future Ready**: Extensible framework for adding news APIs, RSS feeds, and other information sources
- **Graceful Handoffs**: Non-information queries are properly routed to general or tool specialist agents

## Implementation Patterns Used

- **Specialized Agent Pattern**: Domain-specific agent extending base functionality
- **Enhanced Prompting Pattern**: Information-optimized system prompts and instructions
- **Smart Handoff Pattern**: Capability-based routing with fallback logic
- **Response Formatting Pattern**: Domain-specific response enhancement and formatting
- **Tool Clustering Pattern**: Grouping related tools (weather, search, news) under specialized agent

## Testing and Validation

Created comprehensive test suite [`test_information_agent.py`](test_information_agent.py:1) covering:

- Agent initialization and configuration
- Weather query routing and processing
- Web search query routing and processing
- News query routing (placeholder)
- Agent handoff scenarios for non-information queries
- Multi-agent service integration validation

## Follow-up Opportunities

1. **Real News API Integration**: Replace NewsTool placeholder with actual news/RSS services
2. **Advanced Information Synthesis**: Combine multiple information sources in single responses
3. **Caching and Optimization**: Implement intelligent caching for frequently requested information
4. **Contextual Enhancement**: Add location awareness and user preference learning
5. **Performance Monitoring**: Track information query success rates and response quality

## Implications

- Enables sophisticated information retrieval through voice interface
- Provides foundation for news, current events, and research capabilities
- Improves user experience with specialized information processing
- Establishes patterns for future domain-specific agents (e.g., research, analysis)
- Maintains backward compatibility while adding advanced multi-agent capabilities

---

## Decision

2025-08-15 19:02:30 - **UtilityAgent Implementation for Multi-Agent Voice System**

## Rationale

Implemented a specialized UtilityAgent to enhance the multi-agent voice system with dedicated mathematical calculation and utility function capabilities. This addresses the need for:

1. **Specialized Mathematical Processing**: Calculator operations, arithmetic expressions, and computational problem-solving require domain-specific expertise
2. **Enhanced Mathematical Reasoning**: Step-by-step calculation breakdowns with clear explanations and verification
3. **Intelligent Query Routing**: Mathematical queries should be routed to agents with specialized mathematical knowledge
4. **Calculator Tool Integration**: Seamless integration with the existing robust calculator tool for safe mathematical operations
5. **Future Extensibility**: Framework for adding unit conversions, text processing, and other utility functions

## Implementation Details

**Core UtilityAgent Features**:

- Extends [`AgentBase`](src/voice_agent/core/multi_agent/agent_base.py) with mathematics-focused capabilities
- Specialized in [`CALCULATIONS`](src/voice_agent/core/multi_agent/agent_base.py:45), [`TOOL_EXECUTION`](src/voice_agent/core/multi_agent/agent_base.py:41), [`CONVERSATION_MEMORY`](src/voice_agent/core/multi_agent/agent_base.py:49), and [`SYSTEM_INFO`](src/voice_agent/core/multi_agent/agent_base.py:48) capabilities
- Enhanced system prompts optimized for mathematical reasoning and problem-solving
- Smart handoff logic to route non-mathematical queries to appropriate agents
- Mathematical computation tracking and performance metrics
- Response formatting with step-by-step explanations and verification context

**Calculator Tool Integration**:

- Uses existing [`CalculatorTool`](src/voice_agent/tools/builtin/calculator.py:21) for safe mathematical operations
- Supports arithmetic, algebraic expressions, and mathematical functions
- AST-based expression evaluation with comprehensive error handling
- Mathematical constants and helper functions (abs, round, min, max, sum)

**Configuration and Routing**:

- Added UtilityAgent configuration to [`MultiAgentConfig`](src/voice_agent/core/config.py:140) with proper tool assignments
- Created high-priority routing rules for mathematical queries in [`default.yaml`](src/voice_agent/config/default.yaml:147)
- Updated multi-agent service to support UtilityAgent creation and initialization
- Enhanced routing patterns to capture various mathematical query types

**Agent Module Structure**:

- Created [`src/voice_agent/agents/utility_agent.py`](src/voice_agent/agents/utility_agent.py:1) with full UtilityAgent implementation
- Updated [`agents/__init__.py`](src/voice_agent/agents/__init__.py:18) to include UtilityAgent exports
- Updated [`multi_agent_service.py`](src/voice_agent/core/multi_agent_service.py:33) to support UtilityAgent initialization

## Key Architectural Benefits

- **Mathematical Expertise**: Specialized agent for mathematical queries improves calculation accuracy and explanation quality
- **Problem-Solving Enhancement**: Mathematical-specific prompting and step-by-step reasoning approach
- **Smart Routing**: Dedicated routing rules ensure mathematical queries reach the specialized agent
- **Tool Specialization**: Focused calculator tool assignment for optimal mathematical performance
- **Future Ready**: Extensible framework for adding unit conversions, text processing, and data transformations
- **Graceful Handoffs**: Non-mathematical queries are properly routed to general or domain-specific agents

## Implementation Patterns Used

- **Specialized Agent Pattern**: Domain-specific agent extending base functionality for mathematical tasks
- **Enhanced Prompting Pattern**: Mathematics-optimized system prompts and reasoning instructions
- **Smart Handoff Pattern**: Capability-based routing with mathematical keyword detection
- **Response Enhancement Pattern**: Mathematical-specific response formatting and verification context
- **Tool Clustering Pattern**: Focused tool assignment (calculator) under specialized mathematical agent

## Testing and Validation

Created comprehensive test suite [`test_utility_agent.py`](test_utility_agent.py:1) covering:

- Agent initialization and configuration validation
- Mathematical capability verification and routing
- Calculator tool integration and functionality
- Multi-agent service integration (with fallback testing)
- Agent handoff scenarios for mathematical vs. non-mathematical queries
- Basic functionality validation in devenv environment

**Test Results**:

- ✅ UtilityAgent basic functionality: Agent initializes correctly with proper capabilities
- ✅ Calculator tool integration: All mathematical operations working correctly (2+3=5, 10\*5=50, (15+5)/4=5.0, 2^8=256)
- ✅ Configuration integration: Routing rules and agent definitions properly configured
- ⚠️ Multi-agent service integration: Some environment dependency issues with embedding models, but core functionality validated

## Follow-up Opportunities

1. **Unit Conversion System**: Add comprehensive unit conversion capabilities (metric/imperial, currency, etc.)
2. **Advanced Mathematical Functions**: Extend calculator with trigonometry, logarithms, statistics
3. **Mathematical Visualization**: Add simple plotting and graph generation for mathematical results
4. **Mathematical Memory**: Implement calculation history and variable storage across conversations
5. **Performance Optimization**: Cache frequently used mathematical constants and results
6. **Mathematical Validation**: Add result verification and alternative calculation methods

## Implications

- Enables sophisticated mathematical problem-solving through voice interface
- Provides foundation for computational and utility function capabilities
- Improves user experience with specialized mathematical processing and clear explanations
- Establishes patterns for future domain-specific agents (e.g., scientific, engineering, financial)
- Maintains backward compatibility while adding advanced multi-agent mathematical capabilities

---

## Decision

2025-08-15 22:20:00 - **"End Program" Voice Command Implementation for Application Exit**

## Rationale

Added missing voice command functionality to enable users to quit the voice agent application using spoken commands. While the application had proper quit functionality via Ctrl+Q and existing voice command infrastructure, there was no voice command to trigger application exit, creating an inconsistency in the hands-free user experience.

## Implementation Details

**Voice Command Detection Enhancement**:

- Extended [`detect_voice_command()`](src/voice_agent/ui/tui_app.py:1466) method with comprehensive exit command patterns:
  - Primary commands: "end program", "quit", "exit", "shutdown"
  - Extended commands: "close application", "terminate", "quit application", "exit application", "shut down", "close program", "terminate program"
  - Returns normalized command key "quit_app" when these patterns are detected
  - Uses existing pattern matching logic with suffix/prefix tolerance for natural speech variations

**Voice Command Handling Integration**:

- Added "quit_app" command case to [`handle_voice_command()`](src/voice_agent/ui/tui_app.py:1587) method
- Displays user-friendly system message: "(Quitting application via voice command...)"
- Calls existing [`action_quit_app()`](src/voice_agent/ui/tui_app.py:1182) method for proper cleanup
- Uses `asyncio.create_task()` to handle the async quit action properly
- Maintains consistency with existing voice command handling patterns

## Key Architectural Benefits

- **Hands-Free Consistency**: Completes the voice command ecosystem by adding exit functionality
- **Reuses Existing Infrastructure**: Leverages established quit mechanism (`action_quit_app()`) ensuring proper cleanup
- **Natural Language Support**: Multiple command variations accommodate different user speech patterns
- **Safe Integration**: Uses existing voice command detection and handling patterns, maintaining system stability
- **User Experience**: Provides immediate feedback before initiating quit process

## Implementation Patterns Used

- **Command Pattern**: Normalized command keys with centralized handling logic
- **Adapter Pattern**: Voice commands bridge to existing keyboard-based actions
- **Existing Infrastructure Reuse**: Calls established quit method rather than duplicating cleanup logic
- **Async Task Management**: Proper handling of async quit action within voice command context

## Testing and Validation

Implementation maintains compatibility with existing voice command system:

- Voice command detection follows established pattern matching approach
- Command handling integrates seamlessly with existing command types (dictation, privacy mode)
- Quit functionality uses proven cleanup path via `action_quit_app()` method
- No breaking changes to existing voice command functionality

## Implications

- Enables complete hands-free operation including application exit
- Provides consistent user experience across all application functions
- Maintains system reliability through established quit mechanism reuse
- Completes the voice command feature set for core application operations
- Sets pattern for future voice command additions requiring system-level actions

---
