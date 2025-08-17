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

    Audio processing and sensitivity:
    - energy_threshold: Minimum RMS amplitude to consider as potential speech
    - level_meter_scale: Scale factor for audio level meter display

    Barge-in / interruption settings:
    - barge_in_enabled: Allow user to interrupt TTS playback by speaking
    - barge_in_energy_threshold: Audio level threshold for barge-in detection
    - barge_in_consecutive_frames: Number of consecutive frames needed for barge-in

    Audio feedback prevention:
    - feedback_prevention_enabled: Disable microphone during TTS playback
    - buffer_clear_on_playback: Clear input buffer when starting TTS
    - double_buffer_clear: Perform additional buffer clear if experiencing feedback
    """

    # Device configuration
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 16000
    chunk_size: int = 1024

    # VAD / speech detection parameters
    vad_aggressiveness: int = 1
    min_speech_frames: int = 5
    max_silence_frames: int = 50
    speech_detection_cooldown: float = 1.0

    # Audio processing and sensitivity
    energy_threshold: float = 800.0
    level_meter_scale: float = 12000.0

    # Barge-in / interruption settings
    barge_in_enabled: bool = True
    barge_in_energy_threshold: float = 0.28
    barge_in_consecutive_frames: int = 5

    # Audio feedback prevention
    feedback_prevention_enabled: bool = True
    buffer_clear_on_playback: bool = True
    double_buffer_clear: bool = False


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


class WorkflowOrchestrationConfig(BaseModel):
    """Workflow orchestration configuration."""

    enabled: bool = True
    max_concurrent_workflows: int = 3
    default_timeout: int = 300  # seconds
    enable_parallel_execution: bool = True
    enable_pipeline_execution: bool = True
    task_dependency_timeout: int = 60
    workflow_retry_attempts: int = 2
    enable_workflow_monitoring: bool = True
    workflow_history_limit: int = 100


class InterAgentCommunicationConfig(BaseModel):
    """Inter-agent communication configuration."""

    enabled: bool = True
    message_queue_size: int = 1000
    collaboration_timeout: int = 120
    broadcast_enabled: bool = True
    priority_messaging: bool = True
    communication_retry_attempts: int = 3
    channel_buffer_size: int = 500
    enable_message_persistence: bool = False


class EnhancedDelegationConfig(BaseModel):
    """Enhanced delegation configuration."""

    enabled: bool = True
    patterns: List[str] = Field(
        default_factory=lambda: [
            "capability_based",
            "load_balanced",
            "expertise_weighted",
            "collaborative",
            "hierarchical",
            "consensus",
        ]
    )
    consensus_threshold: float = 0.6
    collaboration_min_agents: int = 2
    expertise_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "weather_info": 0.9,
            "calculations": 0.95,
            "file_operations": 0.85,
            "web_search": 0.8,
            "calendar_management": 0.9,
            "task_planning": 0.85,
        }
    )
    load_balancing_factor: float = 0.3
    delegation_timeout: int = 45


class ContextPreservationConfig(BaseModel):
    """Context preservation configuration."""

    enhanced_handoff: bool = True
    context_compression: bool = True
    handoff_metadata: bool = True
    preserve_tool_states: bool = True
    context_validation: bool = True
    max_context_age_hours: int = 48
    context_slice_size: int = 15
    enable_rag_integration: bool = True


class AdvancedFeaturesConfig(BaseModel):
    """Advanced multi-agent features configuration."""

    status_reporting: bool = True
    result_aggregation: bool = True
    collaborative_reasoning: bool = True
    workflow_monitoring: bool = True
    performance_metrics: bool = True
    agent_learning: bool = False  # Future feature
    adaptive_routing: bool = False  # Future feature


class LoadBalancingConfig(BaseModel):
    """Agent load balancing and resource allocation configuration."""

    enabled: bool = True
    algorithm: str = "round_robin"  # round_robin, least_loaded, capability_weighted
    max_agent_load: int = 10
    load_rebalance_interval: int = 30  # seconds
    resource_monitoring: bool = True
    auto_scaling: bool = False  # Future feature
    priority_queuing: bool = True


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    enabled: bool = True
    agent_authentication: bool = False  # Future feature
    message_encryption: bool = False  # Future feature
    audit_logging: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    trusted_agents_only: bool = False
    sandbox_tools: bool = True


class MonitoringConfig(BaseModel):
    """Logging and monitoring configuration."""

    enhanced_logging: bool = True
    performance_tracking: bool = True
    metrics_collection: bool = True
    health_checks: bool = True
    status_reporting_interval: int = 60  # seconds
    log_retention_days: int = 30
    metrics_export_format: str = "json"  # json, prometheus
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "response_time_ms": 5000.0,
            "error_rate_percent": 5.0,
            "agent_availability_percent": 95.0,
        }
    )


class PerformanceTuningConfig(BaseModel):
    """Performance tuning parameters for multi-agent workflows."""

    agent_pool_size: int = 5
    message_batch_size: int = 10
    async_task_limit: int = 100
    memory_limit_mb: int = 1024
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    connection_pool_size: int = 20
    request_timeout_seconds: int = 30
    retry_backoff_factor: float = 1.5


class MultiAgentConfig(BaseModel):
    """Multi-agent system configuration."""

    enabled: bool = False  # Feature flag for multi-agent system
    default_agent: str = "general_agent"
    routing_strategy: str = (
        "hybrid"  # hybrid, rules_only, embeddings_only, llm_fallback
    )
    confidence_threshold: float = 0.7
    max_concurrent_agents: int = 5
    context_sharing_enabled: bool = True
    context_window_size: int = 4000
    embedding_model: str = "nomic-embed-text"

    # Enhanced multi-agent features
    workflow_orchestration: WorkflowOrchestrationConfig = Field(
        default_factory=WorkflowOrchestrationConfig
    )
    inter_agent_communication: InterAgentCommunicationConfig = Field(
        default_factory=InterAgentCommunicationConfig
    )
    enhanced_delegation: EnhancedDelegationConfig = Field(
        default_factory=EnhancedDelegationConfig
    )
    context_preservation: ContextPreservationConfig = Field(
        default_factory=ContextPreservationConfig
    )
    advanced_features: AdvancedFeaturesConfig = Field(
        default_factory=AdvancedFeaturesConfig
    )
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance_tuning: PerformanceTuningConfig = Field(
        default_factory=PerformanceTuningConfig
    )

    # Agent definitions
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "general_agent": {
                "type": "GeneralAgent",
                "capabilities": [
                    "general_chat",
                    "tool_execution",
                    "conversation_memory",
                ],
                "tools": ["calculator", "weather", "file_ops"],
                "system_prompt": "You are a helpful general-purpose AI assistant.",
                "max_concurrent_tasks": 5,
                "timeout_seconds": 30.0,
            },
            "tool_specialist": {
                "type": "ToolSpecialistAgent",
                "capabilities": [
                    "tool_execution",
                    "file_operations",
                    "calculations",
                    "system_info",
                ],
                "tools": ["calculator", "file_ops", "weather", "web_search"],
                "system_prompt": "You are a specialist in tool execution and file operations. Be precise and thorough.",
                "max_concurrent_tasks": 3,
                "timeout_seconds": 60.0,
            },
            "information_agent": {
                "type": "InformationAgent",
                "capabilities": [
                    "weather_info",
                    "web_search",
                    "news_info",
                    "tool_execution",
                    "conversation_memory",
                ],
                "tools": ["weather", "web_search", "news"],
                "system_prompt": "You are an information specialist focused on retrieving and presenting accurate, up-to-date information from various sources.",
                "max_concurrent_tasks": 4,
                "timeout_seconds": 45.0,
            },
            "productivity_agent": {
                "type": "ProductivityAgent",
                "capabilities": [
                    "file_operations",
                    "calendar_management",
                    "tool_execution",
                    "system_info",
                    "conversation_memory",
                    "task_planning",
                ],
                "tools": ["file_ops", "calendar", "calculator"],
                "system_prompt": "You are a ProductivityAgent specialized in productivity and organization tasks, including file operations, calendar management, and task planning.",
                "max_concurrent_tasks": 4,
                "timeout_seconds": 45.0,
            },
            "utility_agent": {
                "type": "UtilityAgent",
                "capabilities": [
                    "calculations",
                    "tool_execution",
                    "conversation_memory",
                    "system_info",
                ],
                "tools": ["calculator"],
                "system_prompt": "You are a UtilityAgent specialized in mathematical calculations and utility functions. Focus on precision, accuracy, and clear explanations of mathematical reasoning.",
                "max_concurrent_tasks": 4,
                "timeout_seconds": 30.0,
            },
        }
    )

    # Routing rules configuration
    routing_rules: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "weather_requests",
                "target_agent": "information_agent",
                "patterns": [
                    "weather",
                    "forecast",
                    "temperature",
                    "rain",
                    "snow",
                    "storm",
                    "sunny",
                    "cloudy",
                    "climate",
                ],
                "capabilities": ["weather_info"],
                "priority": 1,
                "confidence": 0.9,
            },
            {
                "name": "web_search_requests",
                "target_agent": "information_agent",
                "patterns": [
                    "search",
                    "find",
                    "look up",
                    "google",
                    "research",
                    "information about",
                    "tell me about",
                ],
                "capabilities": ["web_search"],
                "priority": 1,
                "confidence": 0.85,
            },
            {
                "name": "news_requests",
                "target_agent": "information_agent",
                "patterns": [
                    "news",
                    "current events",
                    "latest",
                    "recent",
                    "what happened",
                    "breaking news",
                    "headlines",
                ],
                "capabilities": ["news_info"],
                "priority": 1,
                "confidence": 0.9,
            },
            {
                "name": "utility_calculation_requests",
                "target_agent": "utility_agent",
                "patterns": [
                    "calculate",
                    "compute",
                    "math",
                    "equation",
                    "sum",
                    "multiply",
                    "divide",
                    "subtract",
                    "formula",
                    "solve",
                    "arithmetic",
                    "percentage",
                    "ratio",
                ],
                "capabilities": ["calculations"],
                "priority": 1,
                "confidence": 0.9,
            },
            {
                "name": "complex_calculation_requests",
                "target_agent": "utility_agent",
                "patterns": [
                    "mathematical",
                    "algebra",
                    "calculus",
                    "geometry",
                    "trigonometry",
                    "statistics",
                    "probability",
                    "logarithm",
                    "exponential",
                ],
                "capabilities": ["calculations"],
                "priority": 1,
                "confidence": 0.85,
            },
            {
                "name": "file_operations",
                "target_agent": "tool_specialist",
                "patterns": [
                    "file",
                    "directory",
                    "folder",
                    "path",
                    "save",
                    "load",
                    "read",
                    "write",
                ],
                "capabilities": ["file_operations"],
                "priority": 2,
                "confidence": 0.85,
            },
            {
                "name": "productivity_requests",
                "target_agent": "productivity_agent",
                "patterns": [
                    "schedule",
                    "calendar",
                    "appointment",
                    "event",
                    "meeting",
                    "task",
                    "todo",
                    "organize",
                    "plan",
                    "productivity",
                    "deadline",
                    "reminder",
                ],
                "capabilities": [
                    "calendar_management",
                    "task_planning",
                    "file_operations",
                ],
                "priority": 2,
                "confidence": 0.85,
            },
            {
                "name": "general_conversation",
                "target_agent": "general_agent",
                "patterns": [
                    "hello",
                    "hi",
                    "how are you",
                    "tell me",
                    "explain",
                    "what is",
                    "help",
                    "thanks",
                ],
                "capabilities": ["general_chat"],
                "priority": 8,
                "confidence": 0.6,
            },
        ]
    )


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
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)

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

    def validate_multi_agent_config(self) -> List[str]:
        """
        Validate multi-agent configuration and return list of issues.

        Returns:
            List of validation error messages
        """
        issues = []

        if not self.multi_agent.enabled:
            return issues  # Skip validation if multi-agent is disabled

        # Validate agent definitions
        if not self.multi_agent.agents:
            issues.append("No agents defined in multi-agent configuration")
        else:
            # Check if default agent exists
            if self.multi_agent.default_agent not in self.multi_agent.agents:
                issues.append(
                    f"Default agent '{self.multi_agent.default_agent}' not found in agent definitions"
                )

            # Validate individual agent configurations
            for agent_id, agent_config in self.multi_agent.agents.items():
                agent_issues = self._validate_agent_config(agent_id, agent_config)
                issues.extend(agent_issues)

        # Validate routing rules
        routing_issues = self._validate_routing_rules()
        issues.extend(routing_issues)

        # Validate performance settings
        perf_issues = self._validate_performance_settings()
        issues.extend(perf_issues)

        # Validate security settings
        security_issues = self._validate_security_settings()
        issues.extend(security_issues)

        return issues

    def _validate_agent_config(
        self, agent_id: str, agent_config: Dict[str, Any]
    ) -> List[str]:
        """Validate individual agent configuration."""
        issues = []

        required_fields = ["type", "capabilities", "tools", "system_prompt"]
        for field in required_fields:
            if field not in agent_config:
                issues.append(f"Agent '{agent_id}' missing required field: {field}")

        # Validate agent type
        valid_types = [
            "GeneralAgent",
            "ToolSpecialistAgent",
            "InformationAgent",
            "ProductivityAgent",
            "UtilityAgent",
        ]
        if agent_config.get("type") not in valid_types:
            issues.append(
                f"Agent '{agent_id}' has invalid type: {agent_config.get('type')}"
            )

        # Validate capabilities
        capabilities = agent_config.get("capabilities", [])
        if not capabilities:
            issues.append(f"Agent '{agent_id}' has no capabilities defined")

        # Validate tools
        tools = agent_config.get("tools", [])
        if not tools:
            issues.append(f"Agent '{agent_id}' has no tools defined")

        # Validate timeouts and limits
        max_tasks = agent_config.get("max_concurrent_tasks", 3)
        if max_tasks < 1 or max_tasks > 20:
            issues.append(
                f"Agent '{agent_id}' max_concurrent_tasks should be between 1 and 20"
            )

        timeout = agent_config.get("timeout_seconds", 30.0)
        if timeout < 5.0 or timeout > 300.0:
            issues.append(
                f"Agent '{agent_id}' timeout_seconds should be between 5 and 300"
            )

        return issues

    def _validate_routing_rules(self) -> List[str]:
        """Validate routing rules configuration."""
        issues = []

        if not self.multi_agent.routing_rules:
            issues.append("No routing rules defined for multi-agent system")
            return issues

        rule_names = set()
        agent_ids = set(self.multi_agent.agents.keys())

        for i, rule in enumerate(self.multi_agent.routing_rules):
            rule_prefix = f"Routing rule {i + 1}"

            # Check required fields
            if "name" not in rule:
                issues.append(f"{rule_prefix} missing 'name' field")
            else:
                if rule["name"] in rule_names:
                    issues.append(f"{rule_prefix} has duplicate name: {rule['name']}")
                rule_names.add(rule["name"])

            if "target_agent" not in rule:
                issues.append(f"{rule_prefix} missing 'target_agent' field")
            elif rule["target_agent"] not in agent_ids:
                issues.append(
                    f"{rule_prefix} targets non-existent agent: {rule['target_agent']}"
                )

            if "patterns" not in rule or not rule["patterns"]:
                issues.append(f"{rule_prefix} missing or empty 'patterns' field")

            # Validate priority and confidence
            priority = rule.get("priority", 5)
            if not isinstance(priority, int) or priority < 1 or priority > 10:
                issues.append(
                    f"{rule_prefix} priority should be an integer between 1 and 10"
                )

            confidence = rule.get("confidence", 0.8)
            if (
                not isinstance(confidence, (int, float))
                or confidence < 0.0
                or confidence > 1.0
            ):
                issues.append(
                    f"{rule_prefix} confidence should be a number between 0.0 and 1.0"
                )

        return issues

    def _validate_performance_settings(self) -> List[str]:
        """Validate performance tuning settings."""
        issues = []

        perf = self.multi_agent.performance_tuning

        # Validate pool sizes and limits
        if perf.agent_pool_size < 1 or perf.agent_pool_size > 50:
            issues.append("agent_pool_size should be between 1 and 50")

        if perf.message_batch_size < 1 or perf.message_batch_size > 100:
            issues.append("message_batch_size should be between 1 and 100")

        if perf.memory_limit_mb < 256 or perf.memory_limit_mb > 8192:
            issues.append("memory_limit_mb should be between 256 and 8192")

        if perf.request_timeout_seconds < 5 or perf.request_timeout_seconds > 300:
            issues.append("request_timeout_seconds should be between 5 and 300")

        return issues

    def _validate_security_settings(self) -> List[str]:
        """Validate security settings."""
        issues = []

        security = self.multi_agent.security

        if (
            security.max_requests_per_minute < 1
            or security.max_requests_per_minute > 10000
        ):
            issues.append("max_requests_per_minute should be between 1 and 10000")

        return issues

    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run comprehensive health checks on the configuration.

        Returns:
            Dictionary with health check results
        """
        import datetime

        health_status = {
            "overall_health": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "checks": {},
        }

        # Multi-agent validation
        multi_agent_issues = self.validate_multi_agent_config()
        health_status["checks"]["multi_agent_validation"] = {
            "status": "pass" if not multi_agent_issues else "fail",
            "issues": multi_agent_issues,
        }

        # Configuration completeness check
        completeness_issues = self._check_configuration_completeness()
        health_status["checks"]["configuration_completeness"] = {
            "status": "pass" if not completeness_issues else "warning",
            "issues": completeness_issues,
        }

        # Tool availability check
        tool_issues = self._check_tool_availability()
        health_status["checks"]["tool_availability"] = {
            "status": "pass" if not tool_issues else "warning",
            "issues": tool_issues,
        }

        # Resource limits check
        resource_issues = self._check_resource_limits()
        health_status["checks"]["resource_limits"] = {
            "status": "pass" if not resource_issues else "warning",
            "issues": resource_issues,
        }

        # Determine overall health
        failed_checks = [
            check
            for check in health_status["checks"].values()
            if check["status"] == "fail"
        ]
        warning_checks = [
            check
            for check in health_status["checks"].values()
            if check["status"] == "warning"
        ]

        if failed_checks:
            health_status["overall_health"] = "unhealthy"
        elif warning_checks:
            health_status["overall_health"] = "degraded"

        return health_status

    def _check_configuration_completeness(self) -> List[str]:
        """Check if configuration is complete."""
        issues = []

        # Check essential configurations
        if not self.llm.model:
            issues.append("LLM model not specified")

        if not self.tools.enabled:
            issues.append("No tools enabled")

        if self.multi_agent.enabled and not self.multi_agent.embedding_model:
            issues.append("Embedding model not specified for multi-agent system")

        return issues

    def _check_tool_availability(self) -> List[str]:
        """Check if configured tools are available."""
        issues = []

        # This would normally check if tools are actually available
        # For now, just validate the configuration
        enabled_tools = self.tools.enabled
        if not enabled_tools:
            issues.append("No tools are enabled")

        # Check for essential tools
        essential_tools = ["calculator", "weather", "file_ops"]
        missing_essential = [
            tool for tool in essential_tools if tool not in enabled_tools
        ]
        if missing_essential:
            issues.append(f"Missing essential tools: {', '.join(missing_essential)}")

        return issues

    def _check_resource_limits(self) -> List[str]:
        """Check resource limit configurations."""
        issues = []

        if self.multi_agent.enabled:
            # Check if concurrent agents limit is reasonable
            if self.multi_agent.max_concurrent_agents > 20:
                issues.append(
                    "max_concurrent_agents is very high (>20), may cause resource issues"
                )

            # Check workflow limits
            wf_config = self.multi_agent.workflow_orchestration
            if wf_config.max_concurrent_workflows > 10:
                issues.append(
                    "max_concurrent_workflows is very high (>10), may cause resource issues"
                )

        return issues

    def migrate_from_single_agent(self) -> None:
        """
        Migrate configuration from single-agent to multi-agent setup.
        """
        if self.multi_agent.enabled:
            return  # Already using multi-agent

        logging.getLogger(__name__).info(
            "Migrating configuration from single-agent to multi-agent"
        )

        # Enable multi-agent system
        self.multi_agent.enabled = True

        # Ensure all essential agents are configured
        essential_agents = [
            "general_agent",
            "tool_specialist",
            "information_agent",
            "utility_agent",
            "productivity_agent",
        ]

        for agent_id in essential_agents:
            if agent_id not in self.multi_agent.agents:
                logging.getLogger(__name__).warning(
                    f"Adding default configuration for missing agent: {agent_id}"
                )
                # The default agent configurations are already in the factory function

        # Set reasonable defaults for new multi-agent features
        self.multi_agent.workflow_orchestration.enabled = True
        self.multi_agent.inter_agent_communication.enabled = True
        self.multi_agent.enhanced_delegation.enabled = True
        self.multi_agent.context_preservation.enhanced_handoff = True
        self.multi_agent.advanced_features.status_reporting = True

        # Adjust performance settings for multi-agent
        self.multi_agent.performance_tuning.agent_pool_size = 5
        self.multi_agent.performance_tuning.cache_enabled = True

        logging.getLogger(__name__).info(
            "Single-agent to multi-agent migration completed"
        )

    def create_backup(self) -> Dict[str, Any]:
        """
        Create a backup of the current configuration.

        Returns:
            Configuration backup as dictionary
        """
        import datetime

        return {
            "backup_timestamp": datetime.datetime.utcnow().isoformat(),
            "config_version": "1.0",
            "configuration": self.model_dump(),
        }

    def restore_from_backup(self, backup_data: Dict[str, Any]) -> bool:
        """
        Restore configuration from backup data.

        Args:
            backup_data: Backup data dictionary

        Returns:
            True if restore was successful
        """
        try:
            if "configuration" not in backup_data:
                return False

            config_data = backup_data["configuration"]

            # Restore each section
            for section_name in [
                "audio",
                "stt",
                "tts",
                "llm",
                "tools",
                "conversation",
                "logging",
                "ui",
                "multi_agent",
            ]:
                if section_name in config_data:
                    section_config = config_data[section_name]
                    # Update the section with new values
                    current_section = getattr(self, section_name)
                    for key, value in section_config.items():
                        if hasattr(current_section, key):
                            setattr(current_section, key, value)

            logging.getLogger(__name__).info("Configuration restored from backup")
            return True

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to restore configuration from backup: {e}"
            )
            return False
