# Voice Agent TUI Implementation Plan

## Overview

This document provides a comprehensive implementation plan for adding a Text User Interface (TUI) to the Voice Agent application. The TUI will transform the basic CLI into a modern chat-style interface with real-time visual feedback of the audio pipeline status.

## Implementation Checklist

### Phase 1: Foundation Setup

- [ ] Add Rich and Textual dependencies to project
- [ ] Create `src/voice_agent/ui/` directory structure
- [ ] Implement basic TUI application skeleton
- [ ] Add TUI configuration options to default.yaml
- [ ] Create pipeline state management classes
- [ ] Implement basic status indicator system

### Phase 2: Core UI Components

- [ ] Build StatusBar component with pipeline indicators
- [ ] Create ChatView component with conversation display
- [ ] Implement InputHandler for text input mode
- [ ] Add keyboard shortcut handling system
- [ ] Create message formatting and display system
- [ ] Implement conversation history management

### Phase 3: Integration with VoiceAgent

- [ ] Extend VoiceAgent class to TUIVoiceAgent
- [ ] Add state callback system to existing services
- [ ] Integrate pipeline status updates
- [ ] Implement message queue system
- [ ] Add TUI mode selection logic
- [ ] Test basic TUI functionality

### Phase 4: Advanced Features

- [ ] Implement settings modal (F2)
- [ ] Add help overlay system (F1)
- [ ] Create debug panel with log viewer (F3)
- [ ] Add conversation search functionality
- [ ] Implement message export features
- [ ] Add error notification system

### Phase 5: Polish and Testing

- [ ] Implement comprehensive error handling
- [ ] Add graceful fallback mechanisms
- [ ] Create user documentation
- [ ] Perform accessibility testing
- [ ] Add unit tests for TUI components
- [ ] Integration testing with existing services

## Architecture Design

### Framework Selection: Rich + Textual

**Chosen Framework:** Rich + Textual

- **Rationale:** Modern, async-compatible, excellent documentation
- **Benefits:** Advanced formatting, reactive components, built-in widgets
- **Dependencies:** `rich>=13.0.0`, `textual>=0.20.0`

### Visual Layout Design

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎤 Voice Agent                                         v1.0.0    │
├─────────────────────────────────────────────────────────────────┤
│ 🎤🟢 Listening | 🧠🟡 Processing | 💭⚪ LLM | 🗣️⚪ TTS | 🔊⚪ Audio │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [14:32:15] 🤖 Agent: Let me check the weather for you...       │
│            🔧 Using weather tool                                │
│                                                                 │
│ [14:32:18] 🤖 Agent: The current weather in London is          │
│                     partly cloudy with 18°C.                   │
│                                                                 │
│                      User: What's the weather in London? [14:32:15] 👤 │
│                                                                 │
│                                Calculate 25 * 8 [14:32:45] 👤 │
│                                                                 │
│ [14:32:47] 🤖 Agent: 25 multiplied by 8 equals 200.           │
│                                                                 │
│                                     [🎤 Listening...] [14:33:12] 👤 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Space: Talk | Tab: Text | F1: Help | F2: Settings | Ctrl+C: Quit │
└─────────────────────────────────────────────────────────────────┘
```

**Chat Alignment:**

- **Agent messages:** Left-aligned with timestamp on left
- **User messages:** Right-aligned with timestamp on right
- **Color coding:** Different colors for each speaker for visual distinction

### State Management System

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time

class ComponentState(Enum):
    IDLE = "idle"           # 🔴 Component not active
    INITIALIZING = "init"   # 🟡 Starting up
    READY = "ready"         # 🟢 Ready to use
    ACTIVE = "active"       # 🟢 Currently processing
    ERROR = "error"         # 🔴 Error state
    DISABLED = "disabled"   # ⚫ Intentionally disabled

@dataclass
class PipelineStatus:
    audio_input: ComponentState = ComponentState.IDLE
    stt: ComponentState = ComponentState.IDLE
    llm: ComponentState = ComponentState.IDLE
    tts: ComponentState = ComponentState.IDLE
    audio_output: ComponentState = ComponentState.IDLE

    # Additional context
    current_message: Optional[str] = None
    last_update: float = 0.0
    error_message: Optional[str] = None
```

### Visual Status Indicators

```python
STATUS_INDICATORS = {
    ComponentState.IDLE: {
        "color": "dim_white",
        "icon": "⚪",
        "style": "dim"
    },
    ComponentState.INITIALIZING: {
        "color": "yellow",
        "icon": "🟡",
        "style": "blink"
    },
    ComponentState.READY: {
        "color": "green",
        "icon": "🟢",
        "style": "bold"
    },
    ComponentState.ACTIVE: {
        "color": "bright_green",
        "icon": "🟢",
        "style": "bold blink"
    },
    ComponentState.ERROR: {
        "color": "red",
        "icon": "🔴",
        "style": "bold"
    },
    ComponentState.DISABLED: {
        "color": "black",
        "icon": "⚫",
        "style": "dim"
    }
}

COMPONENT_ICONS = {
    "audio_input": "🎤",
    "stt": "🧠",
    "llm": "💭",
    "tts": "🗣️",
    "audio_output": "🔊"
}
```

## Component Architecture

### File Structure

```
src/voice_agent/ui/
├── __init__.py
├── tui_app.py           # Main Textual app
├── components/
│   ├── __init__.py
│   ├── status_bar.py    # Pipeline status indicators
│   ├── chat_view.py     # Conversation display
│   ├── input_handler.py # Text input widget
│   └── settings_modal.py# Configuration overlay
├── models/
│   ├── __init__.py
│   ├── pipeline_state.py # State management
│   └── chat_message.py   # Message models
└── utils/
    ├── __init__.py
    └── formatting.py     # Text formatting utilities
```

### Core Components

#### 1. TUIVoiceAgent

Extended version of existing VoiceAgent with TUI integration:

```python
class TUIVoiceAgent(VoiceAgent):
    def __init__(self, config, tui_enabled=True):
        super().__init__(config)
        self.tui_enabled = tui_enabled
        self.pipeline_state = PipelineStatus()
        self.message_queue = asyncio.Queue()
        self.tui_app = None

    async def initialize(self):
        await super().initialize()
        if self.tui_enabled:
            await self._initialize_tui()

    async def start(self):
        if self.tui_enabled:
            await self._start_with_tui()
        else:
            await super().start()  # Fallback to original behavior
```

#### 2. StatusBar Component

Real-time pipeline status display:

```python
class StatusBar(Widget):
    def __init__(self):
        super().__init__()
        self.pipeline_state = PipelineStatus()

    def update_component_state(self, component: str, state: ComponentState):
        setattr(self.pipeline_state, component, state)
        self.refresh()

    def render(self) -> RenderResult:
        # Render status indicators with colors and icons
        pass
```

#### 3. ChatView Component

Conversation display with proper alignment:

```python
class ChatView(ScrollView):
    def __init__(self):
        super().__init__()
        self.messages = []

    def add_message(self, role: str, content: str, timestamp: datetime):
        message = ChatMessage(role=role, content=content, timestamp=timestamp)
        self.messages.append(message)
        self._render_message(message)

    def _render_message(self, message: ChatMessage):
        if message.role == "agent":
            # Left-aligned with agent color
            self._render_left_aligned(message, "cyan")
        else:
            # Right-aligned with user color
            self._render_right_aligned(message, "green")
```

### Message Structure

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Literal, Optional

@dataclass
class ChatMessage:
    id: str
    role: Literal["user", "agent", "system", "tool"]
    content: str
    timestamp: datetime
    status: Literal["sending", "sent", "processing", "complete", "error"]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Additional context
    audio_duration: Optional[float] = None
    tool_calls: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
```

## Configuration Integration

### Extended Configuration Schema

```yaml
# Add to default.yaml
ui:
  mode: "tui" # "tui", "cli", "auto"
  tui_enabled: true # Enable TUI support
  show_status_bar: true # Show pipeline status
  show_timestamps: true # Show message timestamps
  color_scheme: "dark" # "dark", "light", "auto"
  animation_speed: "normal" # "slow", "normal", "fast", "off"
  message_buffer_size: 100 # Max messages to keep in history

  # Message display preferences
  agent_color: "cyan" # Color for agent messages
  user_color: "green" # Color for user messages
  system_color: "yellow" # Color for system messages

# CLI-specific options
cli:
  verbose_logging: true # Show detailed logs in CLI mode
  show_debug_info: false # Show technical details

# TUI-specific options
tui:
  refresh_rate: 10 # Hz for status updates
  enable_mouse: true # Mouse support
  show_help_bar: true # Bottom help text
  auto_scroll: true # Auto-scroll to new messages
```

### Mode Selection Logic

```python
def determine_ui_mode(config, args):
    if args.tui:
        return "tui"
    elif args.cli:
        return "cli"
    elif args.no_audio:
        return "tui"  # TUI better for text interaction
    elif config.ui.mode == "auto":
        return "tui" if sys.stdout.isatty() else "cli"
    else:
        return config.ui.mode
```

## User Interaction Design

### Keyboard Shortcuts

```python
KEYBOARD_SHORTCUTS = {
    # Global Controls
    "ctrl+c": "quit_application",
    "ctrl+q": "quick_quit_with_confirmation",
    "f1": "show_help_overlay",
    "question_mark": "show_help_overlay",
    "f2": "show_settings_panel",
    "f10": "toggle_fullscreen",

    # Voice Controls
    "space": "push_to_talk",
    "tab": "toggle_text_input_mode",
    "escape": "cancel_current_operation",
    "ctrl+m": "toggle_microphone",
    "ctrl+s": "toggle_speech_output",

    # Chat Interface
    "up": "scroll_conversation_up",
    "down": "scroll_conversation_down",
    "ctrl+home": "go_to_conversation_start",
    "ctrl+end": "go_to_latest_message",
    "ctrl+l": "clear_conversation_history",
    "ctrl+f": "search_conversation",
    "ctrl+c": "copy_selected_message",

    # Text Input Mode
    "enter": "send_message",
    "shift+enter": "new_line",
    "ctrl+a": "select_all_input",
    "ctrl+z": "undo",
    "ctrl+y": "redo",

    # Status & Debug
    "f3": "toggle_detailed_status_view",
    "f4": "toggle_debug_information",
    "f5": "refresh_restart_components",
    "ctrl+r": "reload_configuration"
}
```

### Interaction Modes

1. **Voice-First Mode** (default): Space for push-to-talk, Tab for text fallback
2. **Text Mode**: Keyboard-driven interaction with voice as backup
3. **Hybrid Mode**: Seamless switching between voice and text

## Integration Strategy

### Phase 1: Optional TUI Feature

- Add `--tui` command line flag
- Implement basic TUI alongside existing CLI
- Maintain full backward compatibility
- Allow users to opt-in to new interface

### Phase 2: TUI as Default

- Make TUI the default interface
- Keep CLI available with `--cli` flag
- Ensure smooth transition for existing users
- Update documentation and examples

### Phase 3: Full Integration

- Deep integration with all VoiceAgent features
- Advanced TUI-specific features
- Potential deprecation of pure CLI mode
- Complete user experience optimization

### Integration Points

```python
# Add state callbacks to existing services
class AudioManager:
    def __init__(self, config, state_callback=None):
        # ... existing code ...
        self._state_callback = state_callback

    async def listen(self):
        self._update_state(ComponentState.ACTIVE, "Listening for speech...")
        # ... existing listen logic ...
        self._update_state(ComponentState.READY, "Ready")

    def _update_state(self, state, message=None):
        if self._state_callback:
            self._state_callback('audio_input', state, message)
```

## Error Handling and Recovery

### Error Display Strategy

```python
class TUIErrorHandler:
    def __init__(self, tui_app):
        self.tui_app = tui_app
        self.error_log = []

    def handle_error(self, error_type, message, component=None):
        # Display user-friendly error in TUI
        if error_type == "audio_device":
            self._show_notification("🎤 Audio device unavailable", "warning")
        elif error_type == "network":
            self._show_notification("🌐 Network error - using fallback", "error")
        elif error_type == "model_loading":
            self._show_notification("🧠 Loading model...", "info")

        # Log detailed error for debugging
        self._log_detailed_error(error_type, message, component)
```

### Error Categories and Responses

1. **Critical Errors**: Show modal dialog, offer restart options
2. **Warnings**: Status bar notification with auto-dismiss
3. **Info Messages**: Temporary status updates
4. **Debug Information**: Available in debug panel only

### Recovery Mechanisms

- **Auto-reconnect**: For network and service interruptions
- **Fallback Engines**: Automatic switching between TTS/STT providers
- **Graceful Degradation**: Fall back to text-only mode if audio fails
- **Component Restart**: Restart individual components without full app restart

## Testing Strategy

### Unit Testing

- [ ] Test individual TUI components in isolation
- [ ] Mock VoiceAgent integration points
- [ ] Test state management system
- [ ] Validate keyboard shortcut handling

### Integration Testing

- [ ] Test TUI with real VoiceAgent instance
- [ ] Verify pipeline status updates
- [ ] Test error handling and recovery
- [ ] Validate message display and formatting

### User Acceptance Testing

- [ ] Test with different terminal environments
- [ ] Verify accessibility features
- [ ] Test keyboard navigation
- [ ] Validate color schemes and themes

## Documentation Requirements

### User Documentation

- [ ] TUI usage guide with screenshots
- [ ] Keyboard shortcut reference
- [ ] Configuration options documentation
- [ ] Troubleshooting guide

### Developer Documentation

- [ ] Architecture overview
- [ ] Component API documentation
- [ ] Integration guide for new features
- [ ] Contributing guidelines for TUI components

## Dependencies and Requirements

### New Dependencies

```
rich>=13.0.0           # Terminal formatting and layout
textual>=0.20.0        # TUI framework
```

### System Requirements

- Terminal with 256-color support
- Minimum terminal size: 80x24 characters
- Unicode support for status icons
- Keyboard input support

## Migration and Compatibility

### Backward Compatibility

- Existing CLI behavior preserved
- All current command-line flags supported
- Configuration file compatibility maintained
- Existing scripts and automation unaffected

### Migration Path

1. **Soft Launch**: TUI available with opt-in flag
2. **Default Switch**: TUI becomes default with CLI fallback
3. **Full Integration**: TUI-optimized features and workflows
4. **Legacy Support**: CLI mode maintained for automation/scripts

## Success Metrics

### User Experience Metrics

- [ ] Improved user engagement with visual feedback
- [ ] Reduced confusion about agent status
- [ ] Faster problem resolution with better error display
- [ ] Increased adoption due to modern interface

### Technical Metrics

- [ ] No performance regression from CLI mode
- [ ] Successful integration with all existing features
- [ ] Stable operation across different terminal environments
- [ ] Maintainable and extensible codebase

---

## Implementation Timeline

**Phase 1 (Foundation)**: 2-3 weeks
**Phase 2 (Core UI)**: 3-4 weeks  
**Phase 3 (Integration)**: 2-3 weeks
**Phase 4 (Advanced Features)**: 3-4 weeks
**Phase 5 (Polish & Testing)**: 2-3 weeks

**Total Estimated Timeline**: 12-17 weeks

---

This implementation plan provides a comprehensive roadmap for adding a modern TUI interface to the Voice Agent while maintaining full compatibility with existing functionality. The checkboxes allow for tracking progress throughout the implementation process.
