# Voice Agent Configuration System

This directory contains the complete configuration system for the Voice Agent multi-agent setup. The configuration system provides comprehensive support for multi-agent workflows, validation, templates, CLI management, and migration utilities.

## Overview

The Voice Agent uses a sophisticated multi-agent architecture that requires careful configuration management. This system provides:

- **Multi-Agent Configuration**: Complete setup for specialized agents (Information, Productivity, Utility, General, Tool Specialist)
- **Routing and Orchestration**: Intelligent task routing and workflow management
- **Performance Tuning**: Load balancing, resource allocation, and optimization settings
- **Security and Monitoring**: Access control, audit logging, and health monitoring
- **Template System**: Pre-configured setups for different deployment scenarios
- **CLI Management**: Command-line tools for configuration operations
- **Migration Support**: Smooth transition from single-agent to multi-agent setups

## File Structure

```
src/voice_agent/config/
├── README.md                 # This documentation
├── default.yaml             # Default configuration with multi-agent setup
├── config.py               # Core configuration models and validation (in core/)
├── templates.py            # Configuration templates for different scenarios
├── cli_manager.py          # Command-line interface for configuration management
└── test_config.py          # Comprehensive test suite
```

## Configuration Files

### Main Configuration (`default.yaml`)

The main configuration file defines all aspects of the voice agent system:

```yaml
# Multi-agent configuration
multi_agent:
  enabled: true
  default_agent: "general"

  # Agent definitions
  agents:
    information:
      enabled: true
      model_config:
        model: "gpt-4"
        temperature: 0.7
      capabilities:
        - "web_search"
        - "knowledge_retrieval"
        - "fact_checking"

    productivity:
      enabled: true
      model_config:
        model: "gpt-4"
        temperature: 0.5
      capabilities:
        - "calendar_management"
        - "email_handling"
        - "task_planning"

  # Routing rules
  routing:
    rules:
      - patterns: ["search", "find", "lookup", "what is"]
        agent: "information"
        priority: 1
      - patterns: ["schedule", "calendar", "meeting", "appointment"]
        agent: "productivity"
        priority: 1

# TTS/STT Configuration
tts:
  provider: "openai"
  voice: "nova"
  speed: 1.0

stt:
  provider: "openai"
  language: "en-US"
```

### Configuration Classes

The configuration system uses Pydantic models for type safety and validation:

- **`Config`**: Main configuration class
- **`MultiAgentConfig`**: Multi-agent specific settings
- **`AgentConfig`**: Individual agent configuration
- **`RoutingConfig`**: Task routing rules
- **`WorkflowOrchestrationConfig`**: Workflow management settings
- **`InterAgentCommunicationConfig`**: Communication parameters
- **`PerformanceTuningConfig`**: Performance optimization
- **`SecurityConfig`**: Security and access control
- **`MonitoringConfig`**: Logging and monitoring

## Templates

The template system provides pre-configured setups for different use cases:

### Available Templates

1. **Basic Template** (`basic`)
   - Minimal multi-agent setup
   - Essential agents only (Information, General)
   - Suitable for simple applications

2. **Advanced Template** (`advanced`)
   - Full multi-agent setup
   - All agents enabled with comprehensive capabilities
   - Enhanced routing and communication
   - Suitable for complex applications

3. **Development Template** (`development`)
   - Optimized for development and testing
   - Debug logging enabled
   - Relaxed security settings
   - Fast response times

4. **Production Template** (`production`)
   - Optimized for production deployment
   - Enhanced security and monitoring
   - Load balancing and caching enabled
   - Comprehensive audit logging

5. **Testing Template** (`testing`)
   - Configured for automated testing
   - Mock integrations
   - Predictable behavior settings

### Using Templates

```python
from voice_agent.config.templates import ConfigurationTemplates

templates = ConfigurationTemplates()

# List available templates
available = templates.list_templates()

# Get a specific template
production_config = templates.get_template("production")

# Get template description
description = templates.get_template_description("production")
```

## CLI Management

The CLI manager provides command-line interface for configuration operations:

### Available Commands

```bash
# Validate configuration
python -m voice_agent.config.cli_manager validate --config config.yaml

# Run health checks
python -m voice_agent.config.cli_manager health --config config.yaml --format json

# List available templates
python -m voice_agent.config.cli_manager template list

# Apply a template
python -m voice_agent.config.cli_manager template apply production --output config.yaml

# Show template configuration
python -m voice_agent.config.cli_manager template show advanced --format yaml

# Migrate from single-agent to multi-agent
python -m voice_agent.config.cli_manager migrate single-to-multi --config config.yaml --backup

# Create configuration backup
python -m voice_agent.config.cli_manager backup --config config.yaml --output backup.json

# Restore from backup
python -m voice_agent.config.cli_manager restore backup.json --config config.yaml

# Set configuration values
python -m voice_agent.config.cli_manager set multi_agent.enabled=true tts.voice=alloy

# Get configuration values
python -m voice_agent.config.cli_manager get multi_agent.enabled tts.voice
```

### CLI Examples

```bash
# Quick setup for production
python -m voice_agent.config.cli_manager template apply production

# Validate and check health
python -m voice_agent.config.cli_manager validate --strict
python -m voice_agent.config.cli_manager health --format json

# Backup before changes
python -m voice_agent.config.cli_manager backup --output pre-change-backup.json

# Make configuration changes
python -m voice_agent.config.cli_manager set \
  multi_agent.workflow_orchestration.max_parallel_tasks=8 \
  multi_agent.performance_tuning.load_balancing.strategy=round_robin
```

## Configuration Validation

The system provides comprehensive validation at multiple levels:

### Validation Types

1. **Schema Validation**: Ensures configuration structure is correct
2. **Multi-Agent Validation**: Validates agent configurations and routing rules
3. **Performance Validation**: Checks performance settings and resource limits
4. **Security Validation**: Validates security configurations and access controls

### Running Validation

```python
from voice_agent.core.config import Config

# Load and validate configuration
config = Config.load("config.yaml")

# Get validation issues
issues = config.validate_multi_agent_config()

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid")
```

## Health Checks

The health check system monitors configuration and system health:

### Health Check Categories

1. **Multi-Agent Validation**: Agent configurations and routing
2. **Configuration Completeness**: Required settings and dependencies
3. **Tool Availability**: External tool and service availability
4. **Resource Limits**: Memory, CPU, and connection limits

### Running Health Checks

```python
from voice_agent.core.config import Config

config = Config.load("config.yaml")
health_results = config.run_health_checks()

print(f"Overall Health: {health_results['overall_health']}")
print("Health Checks:")
for check_name, result in health_results['checks'].items():
    print(f"  {check_name}: {result['status']}")
    if result['issues']:
        for issue in result['issues']:
            print(f"    - {issue}")
```

## Migration

The migration system helps transition from single-agent to multi-agent configurations:

### Migration Process

1. **Backup Creation**: Automatic backup of current configuration
2. **Agent Setup**: Creates default agent configurations
3. **Routing Configuration**: Sets up basic routing rules
4. **Validation**: Ensures migrated configuration is valid

### Performing Migration

```python
from voice_agent.core.config import Config

# Load single-agent configuration
config = Config.load("single_agent_config.yaml")

# Perform migration
config.migrate_from_single_agent()

# Save migrated configuration
config.save("multi_agent_config.yaml")
```

## Advanced Configuration

### Agent Capabilities

Each agent can be configured with specific capabilities:

```yaml
agents:
  information:
    capabilities:
      - "web_search" # Web search functionality
      - "knowledge_retrieval" # RAG-based knowledge retrieval
      - "fact_checking" # Information verification
      - "data_analysis" # Basic data analysis

  productivity:
    capabilities:
      - "calendar_management" # Calendar operations
      - "email_handling" # Email processing
      - "task_planning" # Task and project planning
      - "document_generation" # Document creation
```

### Routing Rules

Complex routing rules can be defined for intelligent task assignment:

```yaml
routing:
  rules:
    - patterns: ["search", "find", "lookup", "research"]
      agent: "information"
      priority: 1
      conditions:
        - type: "context_length"
          operator: "<"
          value: 1000

    - patterns: ["schedule", "calendar", "meeting"]
      agent: "productivity"
      priority: 1
      conditions:
        - type: "user_role"
          operator: "in"
          value: ["manager", "executive"]
```

### Workflow Orchestration

Configure how tasks are orchestrated across agents:

```yaml
workflow_orchestration:
  execution_modes:
    - "sequential" # Tasks executed one after another
    - "parallel" # Tasks executed simultaneously
    - "pipeline" # Output of one task feeds into next

  max_parallel_tasks: 5
  task_timeout: 300
  retry_policy:
    max_retries: 3
    backoff_strategy: "exponential"
```

### Performance Tuning

Optimize performance for your specific use case:

```yaml
performance_tuning:
  load_balancing:
    enabled: true
    strategy: "least_connections" # or "round_robin", "weighted"
    health_check_interval: 30

  resource_allocation:
    max_memory_per_agent: "1GB"
    max_cpu_per_agent: 2
    agent_pool_size: 5

  caching:
    enabled: true
    ttl: 3600
    max_size: "100MB"
```

### Security Configuration

Secure your multi-agent system:

```yaml
security:
  rate_limiting:
    requests_per_minute: 100
    burst_limit: 20

  audit_logging:
    enabled: true
    log_level: "INFO"
    include_request_data: false

  access_control:
    require_authentication: true
    allowed_origins: ["https://yourapp.com"]
```

## Testing

The configuration system includes comprehensive tests:

### Running Tests

```bash
# Run all configuration tests
python -m voice_agent.config.test_config

# Run specific test categories
python -m unittest voice_agent.config.test_config.TestConfigurationSystem
python -m unittest voice_agent.config.test_config.TestConfigurationIntegration
```

### Test Categories

1. **Configuration Loading**: Test YAML loading and parsing
2. **Validation**: Test all validation mechanisms
3. **Health Checks**: Test health monitoring system
4. **Migration**: Test single-agent to multi-agent migration
5. **Templates**: Test template system functionality
6. **CLI Operations**: Test command-line interface
7. **Integration**: End-to-end workflow testing

## Best Practices

### Configuration Management

1. **Use Templates**: Start with appropriate templates for your use case
2. **Validate Early**: Always validate configuration before deployment
3. **Monitor Health**: Set up regular health check monitoring
4. **Backup Regularly**: Create backups before making changes
5. **Version Control**: Keep configuration files in version control

### Performance Optimization

1. **Agent Selection**: Enable only the agents you need
2. **Resource Limits**: Set appropriate resource limits for your environment
3. **Caching**: Enable caching for frequently accessed data
4. **Load Balancing**: Use load balancing for high-traffic scenarios

### Security Considerations

1. **Access Control**: Implement proper authentication and authorization
2. **Rate Limiting**: Protect against abuse with rate limiting
3. **Audit Logging**: Enable comprehensive audit logging
4. **Secret Management**: Use secure secret management for API keys

### Development Workflow

1. **Development Template**: Use development template for local development
2. **Testing Configuration**: Use testing template for automated tests
3. **Staging Environment**: Test production configuration in staging
4. **Gradual Rollout**: Roll out configuration changes gradually

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   - Check YAML syntax
   - Verify all required fields are present
   - Ensure agent capabilities are valid

2. **Health Check Failures**
   - Check external service availability
   - Verify resource limits are not exceeded
   - Review agent configurations

3. **Migration Issues**
   - Ensure backup is created before migration
   - Check single-agent configuration is valid
   - Verify migration completed successfully

4. **Performance Problems**
   - Review resource allocation settings
   - Check load balancing configuration
   - Monitor agent utilization

### Getting Help

- Check the test suite for examples of proper usage
- Review template configurations for reference implementations
- Use CLI validation and health checks to identify issues
- Enable debug logging for detailed troubleshooting information

## Migration Guide

### From Single-Agent to Multi-Agent

1. **Backup Current Configuration**

   ```bash
   python -m voice_agent.config.cli_manager backup --output pre-migration-backup.json
   ```

2. **Run Migration**

   ```bash
   python -m voice_agent.config.cli_manager migrate single-to-multi --backup
   ```

3. **Validate Migrated Configuration**

   ```bash
   python -m voice_agent.config.cli_manager validate --strict
   ```

4. **Test Health**

   ```bash
   python -m voice_agent.config.cli_manager health
   ```

5. **Customize as Needed**
   - Adjust agent capabilities
   - Fine-tune routing rules
   - Configure performance settings

### Upgrading Configuration Schema

When upgrading to newer versions of the configuration schema:

1. **Check Compatibility**: Review changelog for breaking changes
2. **Backup Configuration**: Always backup before upgrading
3. **Run Migration**: Use provided migration tools
4. **Validate Result**: Ensure migrated configuration is valid
5. **Test Functionality**: Verify all features work correctly

This configuration system provides a robust foundation for managing complex multi-agent voice assistant deployments. It balances flexibility with safety, ensuring that your voice agent can scale and adapt to changing requirements while maintaining reliability and performance.
