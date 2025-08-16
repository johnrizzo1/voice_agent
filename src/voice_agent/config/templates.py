"""
Configuration templates for different use cases.

Provides pre-configured templates for basic, advanced, development,
production, and testing scenarios.
"""

from typing import Dict, Any, List


class ConfigurationTemplates:
    """Configuration templates for different deployment scenarios."""

    @staticmethod
    def get_basic_config() -> Dict[str, Any]:
        """Get basic multi-agent configuration for simple use cases."""
        return {
            "multi_agent": {
                "enabled": True,
                "default_agent": "general_agent",
                "routing_strategy": "rules_only",
                "confidence_threshold": 0.8,
                "max_concurrent_agents": 3,
                "context_sharing_enabled": True,
                "context_window_size": 2048,
                "embedding_model": "nomic-embed-text",
                # Simplified workflow orchestration
                "workflow_orchestration": {
                    "enabled": False,
                    "max_concurrent_workflows": 1,
                    "default_timeout": 120,
                    "enable_parallel_execution": False,
                    "enable_pipeline_execution": True,
                    "task_dependency_timeout": 30,
                },
                # Basic communication
                "inter_agent_communication": {
                    "enabled": True,
                    "message_queue_size": 100,
                    "collaboration_timeout": 60,
                    "broadcast_enabled": False,
                    "priority_messaging": False,
                    "communication_retry_attempts": 1,
                },
                # Simple delegation
                "enhanced_delegation": {
                    "enabled": True,
                    "patterns": ["capability_based", "load_balanced"],
                    "consensus_threshold": 0.7,
                    "collaboration_min_agents": 2,
                    "delegation_timeout": 30,
                },
                # Basic context preservation
                "context_preservation": {
                    "enhanced_handoff": True,
                    "context_compression": False,
                    "handoff_metadata": True,
                    "preserve_tool_states": True,
                    "context_validation": False,
                    "max_context_age_hours": 24,
                },
                # Essential features only
                "advanced_features": {
                    "status_reporting": True,
                    "result_aggregation": False,
                    "collaborative_reasoning": False,
                    "workflow_monitoring": False,
                    "performance_metrics": True,
                },
                # Basic load balancing
                "load_balancing": {
                    "enabled": True,
                    "algorithm": "round_robin",
                    "max_agent_load": 5,
                    "load_rebalance_interval": 60,
                    "resource_monitoring": False,
                    "priority_queuing": False,
                },
                # Minimal security
                "security": {
                    "enabled": True,
                    "audit_logging": False,
                    "rate_limiting": False,
                    "sandbox_tools": True,
                },
                # Basic monitoring
                "monitoring": {
                    "enhanced_logging": False,
                    "performance_tracking": True,
                    "metrics_collection": False,
                    "health_checks": True,
                    "status_reporting_interval": 120,
                },
                # Conservative performance settings
                "performance_tuning": {
                    "agent_pool_size": 3,
                    "message_batch_size": 5,
                    "async_task_limit": 20,
                    "memory_limit_mb": 512,
                    "cache_enabled": True,
                    "cache_ttl_seconds": 300,
                    "connection_pool_size": 5,
                    "request_timeout_seconds": 30,
                },
            }
        }

    @staticmethod
    def get_advanced_config() -> Dict[str, Any]:
        """Get advanced multi-agent configuration with all features enabled."""
        return {
            "multi_agent": {
                "enabled": True,
                "default_agent": "general_agent",
                "routing_strategy": "hybrid",
                "confidence_threshold": 0.7,
                "max_concurrent_agents": 8,
                "context_sharing_enabled": True,
                "context_window_size": 6000,
                "embedding_model": "nomic-embed-text",
                # Full workflow orchestration
                "workflow_orchestration": {
                    "enabled": True,
                    "max_concurrent_workflows": 5,
                    "default_timeout": 600,
                    "enable_parallel_execution": True,
                    "enable_pipeline_execution": True,
                    "task_dependency_timeout": 120,
                    "workflow_retry_attempts": 3,
                    "enable_workflow_monitoring": True,
                    "workflow_history_limit": 200,
                },
                # Advanced communication
                "inter_agent_communication": {
                    "enabled": True,
                    "message_queue_size": 2000,
                    "collaboration_timeout": 300,
                    "broadcast_enabled": True,
                    "priority_messaging": True,
                    "communication_retry_attempts": 3,
                    "channel_buffer_size": 1000,
                    "enable_message_persistence": True,
                },
                # All delegation patterns
                "enhanced_delegation": {
                    "enabled": True,
                    "patterns": [
                        "capability_based",
                        "load_balanced",
                        "expertise_weighted",
                        "collaborative",
                        "hierarchical",
                        "consensus",
                    ],
                    "consensus_threshold": 0.6,
                    "collaboration_min_agents": 2,
                    "expertise_weights": {
                        "weather_info": 0.9,
                        "calculations": 0.95,
                        "file_operations": 0.85,
                        "web_search": 0.8,
                        "calendar_management": 0.9,
                        "task_planning": 0.85,
                    },
                    "load_balancing_factor": 0.3,
                    "delegation_timeout": 60,
                },
                # Advanced context preservation
                "context_preservation": {
                    "enhanced_handoff": True,
                    "context_compression": True,
                    "handoff_metadata": True,
                    "preserve_tool_states": True,
                    "context_validation": True,
                    "max_context_age_hours": 72,
                    "context_slice_size": 20,
                    "enable_rag_integration": True,
                },
                # All advanced features
                "advanced_features": {
                    "status_reporting": True,
                    "result_aggregation": True,
                    "collaborative_reasoning": True,
                    "workflow_monitoring": True,
                    "performance_metrics": True,
                    "agent_learning": False,  # Future feature
                    "adaptive_routing": False,  # Future feature
                },
                # Advanced load balancing
                "load_balancing": {
                    "enabled": True,
                    "algorithm": "capability_weighted",
                    "max_agent_load": 15,
                    "load_rebalance_interval": 15,
                    "resource_monitoring": True,
                    "auto_scaling": False,
                    "priority_queuing": True,
                },
                # Enhanced security
                "security": {
                    "enabled": True,
                    "agent_authentication": False,
                    "message_encryption": False,
                    "audit_logging": True,
                    "rate_limiting": True,
                    "max_requests_per_minute": 200,
                    "trusted_agents_only": False,
                    "sandbox_tools": True,
                },
                # Comprehensive monitoring
                "monitoring": {
                    "enhanced_logging": True,
                    "performance_tracking": True,
                    "metrics_collection": True,
                    "health_checks": True,
                    "status_reporting_interval": 30,
                    "log_retention_days": 60,
                    "metrics_export_format": "json",
                    "alert_thresholds": {
                        "response_time_ms": 3000.0,
                        "error_rate_percent": 3.0,
                        "agent_availability_percent": 98.0,
                    },
                },
                # High-performance settings
                "performance_tuning": {
                    "agent_pool_size": 10,
                    "message_batch_size": 20,
                    "async_task_limit": 200,
                    "memory_limit_mb": 2048,
                    "cache_enabled": True,
                    "cache_ttl_seconds": 600,
                    "connection_pool_size": 50,
                    "request_timeout_seconds": 60,
                    "retry_backoff_factor": 2.0,
                },
            }
        }

    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Get development configuration with debugging features."""
        return {
            "multi_agent": {
                "enabled": True,
                "default_agent": "general_agent",
                "routing_strategy": "hybrid",
                "confidence_threshold": 0.6,  # Lower for testing
                "max_concurrent_agents": 3,
                "context_sharing_enabled": True,
                "context_window_size": 2048,
                "embedding_model": "nomic-embed-text",
                # Development workflow settings
                "workflow_orchestration": {
                    "enabled": True,
                    "max_concurrent_workflows": 2,
                    "default_timeout": 60,  # Shorter for dev
                    "enable_parallel_execution": True,
                    "enable_pipeline_execution": True,
                    "task_dependency_timeout": 30,
                    "workflow_retry_attempts": 1,  # Less retries for faster feedback
                    "enable_workflow_monitoring": True,
                    "workflow_history_limit": 50,
                },
                # Development communication
                "inter_agent_communication": {
                    "enabled": True,
                    "message_queue_size": 200,
                    "collaboration_timeout": 60,
                    "broadcast_enabled": True,
                    "priority_messaging": True,
                    "communication_retry_attempts": 1,
                    "channel_buffer_size": 100,
                    "enable_message_persistence": False,  # Reduce I/O in dev
                },
                # All delegation patterns for testing
                "enhanced_delegation": {
                    "enabled": True,
                    "patterns": [
                        "capability_based",
                        "load_balanced",
                        "expertise_weighted",
                        "collaborative",
                    ],
                    "consensus_threshold": 0.5,  # Lower for testing
                    "collaboration_min_agents": 2,
                    "delegation_timeout": 30,
                },
                # Enhanced debugging context
                "context_preservation": {
                    "enhanced_handoff": True,
                    "context_compression": False,  # Keep full context for debugging
                    "handoff_metadata": True,
                    "preserve_tool_states": True,
                    "context_validation": True,
                    "max_context_age_hours": 12,  # Shorter for dev cycles
                    "context_slice_size": 25,  # More context for debugging
                    "enable_rag_integration": True,
                },
                # All features for testing
                "advanced_features": {
                    "status_reporting": True,
                    "result_aggregation": True,
                    "collaborative_reasoning": True,
                    "workflow_monitoring": True,
                    "performance_metrics": True,
                },
                # Development load balancing
                "load_balancing": {
                    "enabled": True,
                    "algorithm": "round_robin",  # Predictable for debugging
                    "max_agent_load": 3,
                    "load_rebalance_interval": 30,
                    "resource_monitoring": True,
                    "priority_queuing": True,
                },
                # Relaxed security for development
                "security": {
                    "enabled": True,
                    "audit_logging": True,
                    "rate_limiting": False,  # No rate limiting in dev
                    "sandbox_tools": True,
                },
                # Verbose monitoring for debugging
                "monitoring": {
                    "enhanced_logging": True,
                    "performance_tracking": True,
                    "metrics_collection": True,
                    "health_checks": True,
                    "status_reporting_interval": 15,  # Frequent updates
                    "log_retention_days": 7,  # Short retention in dev
                    "metrics_export_format": "json",
                    "alert_thresholds": {
                        "response_time_ms": 10000.0,  # Relaxed for debugging
                        "error_rate_percent": 10.0,
                        "agent_availability_percent": 90.0,
                    },
                },
                # Development performance settings
                "performance_tuning": {
                    "agent_pool_size": 3,
                    "message_batch_size": 5,
                    "async_task_limit": 50,
                    "memory_limit_mb": 1024,
                    "cache_enabled": False,  # Disable cache for testing
                    "cache_ttl_seconds": 60,
                    "connection_pool_size": 10,
                    "request_timeout_seconds": 15,  # Shorter timeouts for faster feedback
                    "retry_backoff_factor": 1.0,
                },
            },
            # Development-specific logging
            "logging": {
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Get production configuration optimized for stability and performance."""
        return {
            "multi_agent": {
                "enabled": True,
                "default_agent": "general_agent",
                "routing_strategy": "hybrid",
                "confidence_threshold": 0.8,  # Higher for production stability
                "max_concurrent_agents": 6,
                "context_sharing_enabled": True,
                "context_window_size": 4000,
                "embedding_model": "nomic-embed-text",
                # Production workflow settings
                "workflow_orchestration": {
                    "enabled": True,
                    "max_concurrent_workflows": 3,
                    "default_timeout": 300,
                    "enable_parallel_execution": True,
                    "enable_pipeline_execution": True,
                    "task_dependency_timeout": 60,
                    "workflow_retry_attempts": 3,
                    "enable_workflow_monitoring": True,
                    "workflow_history_limit": 1000,
                },
                # Production communication
                "inter_agent_communication": {
                    "enabled": True,
                    "message_queue_size": 1000,
                    "collaboration_timeout": 180,
                    "broadcast_enabled": True,
                    "priority_messaging": True,
                    "communication_retry_attempts": 3,
                    "channel_buffer_size": 500,
                    "enable_message_persistence": True,
                },
                # Stable delegation patterns
                "enhanced_delegation": {
                    "enabled": True,
                    "patterns": [
                        "capability_based",
                        "load_balanced",
                        "expertise_weighted",
                    ],
                    "consensus_threshold": 0.7,
                    "collaboration_min_agents": 2,
                    "delegation_timeout": 60,
                },
                # Production context preservation
                "context_preservation": {
                    "enhanced_handoff": True,
                    "context_compression": True,  # Save memory in production
                    "handoff_metadata": True,
                    "preserve_tool_states": True,
                    "context_validation": True,
                    "max_context_age_hours": 48,
                    "context_slice_size": 15,
                    "enable_rag_integration": True,
                },
                # Essential features for production
                "advanced_features": {
                    "status_reporting": True,
                    "result_aggregation": True,
                    "collaborative_reasoning": False,  # Disable for performance
                    "workflow_monitoring": True,
                    "performance_metrics": True,
                },
                # Production load balancing
                "load_balancing": {
                    "enabled": True,
                    "algorithm": "least_loaded",
                    "max_agent_load": 10,
                    "load_rebalance_interval": 30,
                    "resource_monitoring": True,
                    "priority_queuing": True,
                },
                # Production security
                "security": {
                    "enabled": True,
                    "audit_logging": True,
                    "rate_limiting": True,
                    "max_requests_per_minute": 120,
                    "sandbox_tools": True,
                },
                # Production monitoring
                "monitoring": {
                    "enhanced_logging": False,  # Reduce log volume
                    "performance_tracking": True,
                    "metrics_collection": True,
                    "health_checks": True,
                    "status_reporting_interval": 60,
                    "log_retention_days": 90,
                    "metrics_export_format": "json",
                    "alert_thresholds": {
                        "response_time_ms": 5000.0,
                        "error_rate_percent": 2.0,
                        "agent_availability_percent": 99.0,
                    },
                },
                # Production performance settings
                "performance_tuning": {
                    "agent_pool_size": 8,
                    "message_batch_size": 15,
                    "async_task_limit": 100,
                    "memory_limit_mb": 1536,
                    "cache_enabled": True,
                    "cache_ttl_seconds": 900,  # Longer cache in production
                    "connection_pool_size": 30,
                    "request_timeout_seconds": 45,
                    "retry_backoff_factor": 1.5,
                },
            },
            # Production logging
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    @staticmethod
    def get_testing_config() -> Dict[str, Any]:
        """Get testing configuration optimized for test suites."""
        return {
            "multi_agent": {
                "enabled": True,
                "default_agent": "general_agent",
                "routing_strategy": "rules_only",  # Predictable for tests
                "confidence_threshold": 0.9,  # High confidence for test stability
                "max_concurrent_agents": 2,  # Minimal for tests
                "context_sharing_enabled": True,
                "context_window_size": 1024,  # Small for tests
                "embedding_model": "nomic-embed-text",
                # Minimal workflow for tests
                "workflow_orchestration": {
                    "enabled": False,  # Disable complex workflows in tests
                    "max_concurrent_workflows": 1,
                    "default_timeout": 30,
                    "enable_parallel_execution": False,
                    "enable_pipeline_execution": True,
                    "task_dependency_timeout": 10,
                },
                # Minimal communication for tests
                "inter_agent_communication": {
                    "enabled": True,
                    "message_queue_size": 50,
                    "collaboration_timeout": 30,
                    "broadcast_enabled": False,
                    "priority_messaging": False,
                    "communication_retry_attempts": 1,
                    "channel_buffer_size": 20,
                    "enable_message_persistence": False,
                },
                # Simple delegation for tests
                "enhanced_delegation": {
                    "enabled": True,
                    "patterns": [
                        "capability_based"
                    ],  # Single pattern for predictability
                    "consensus_threshold": 0.8,
                    "collaboration_min_agents": 2,
                    "delegation_timeout": 15,
                },
                # Minimal context for tests
                "context_preservation": {
                    "enhanced_handoff": True,
                    "context_compression": False,
                    "handoff_metadata": False,
                    "preserve_tool_states": False,
                    "context_validation": False,
                    "max_context_age_hours": 1,
                    "context_slice_size": 5,
                    "enable_rag_integration": False,
                },
                # Essential features only for tests
                "advanced_features": {
                    "status_reporting": False,
                    "result_aggregation": False,
                    "collaborative_reasoning": False,
                    "workflow_monitoring": False,
                    "performance_metrics": False,
                },
                # No load balancing for tests
                "load_balancing": {
                    "enabled": False,
                    "algorithm": "round_robin",
                    "max_agent_load": 1,
                    "load_rebalance_interval": 60,
                    "resource_monitoring": False,
                    "priority_queuing": False,
                },
                # Minimal security for tests
                "security": {
                    "enabled": False,
                    "audit_logging": False,
                    "rate_limiting": False,
                    "sandbox_tools": True,
                },
                # Minimal monitoring for tests
                "monitoring": {
                    "enhanced_logging": False,
                    "performance_tracking": False,
                    "metrics_collection": False,
                    "health_checks": False,
                    "status_reporting_interval": 300,
                },
                # Minimal performance for tests
                "performance_tuning": {
                    "agent_pool_size": 2,
                    "message_batch_size": 1,
                    "async_task_limit": 10,
                    "memory_limit_mb": 256,
                    "cache_enabled": False,
                    "cache_ttl_seconds": 60,
                    "connection_pool_size": 2,
                    "request_timeout_seconds": 10,
                    "retry_backoff_factor": 1.0,
                },
            },
            # Test logging
            "logging": {
                "level": "WARNING",  # Minimal logging in tests
                "format": "%(levelname)s - %(message)s",
            },
        }

    @classmethod
    def get_template(cls, template_name: str) -> Dict[str, Any]:
        """Get a configuration template by name."""
        templates = {
            "basic": cls.get_basic_config,
            "advanced": cls.get_advanced_config,
            "development": cls.get_development_config,
            "production": cls.get_production_config,
            "testing": cls.get_testing_config,
        }

        if template_name not in templates:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(templates.keys())}"
            )

        return templates[template_name]()

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available configuration templates."""
        return ["basic", "advanced", "development", "production", "testing"]

    @classmethod
    def get_template_description(cls, template_name: str) -> str:
        """Get description of a configuration template."""
        descriptions = {
            "basic": "Simple multi-agent setup with essential features only",
            "advanced": "Full-featured multi-agent setup with all capabilities enabled",
            "development": "Development-optimized setup with debugging features and verbose logging",
            "production": "Production-optimized setup focused on stability and performance",
            "testing": "Minimal setup optimized for automated test suites",
        }

        return descriptions.get(template_name, "No description available")
