"""
Test suite for Voice Agent Configuration System.

Comprehensive tests for multi-agent configuration, validation,
templates, CLI management, and migration functionality.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import yaml

from voice_agent.core.config import Config
from voice_agent.config.templates import ConfigurationTemplates
from voice_agent.config.cli_manager import CLIConfigManager


class TestConfigurationSystem(unittest.TestCase):
    """Test suite for configuration system components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.yaml"
        self.templates = ConfigurationTemplates()

        # Create basic test configuration
        self.basic_config = {
            "multi_agent": {
                "enabled": True,
                "agents": {
                    "information": {
                        "enabled": True,
                        "model_config": {"model": "gpt-4", "temperature": 0.7},
                    }
                },
            },
            "tts": {"provider": "openai", "voice": "nova"},
            "stt": {"provider": "openai"},
        }

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_config(self, config_dict: Dict[str, Any]) -> Path:
        """Save configuration dictionary to test file."""
        with open(self.config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        return self.config_file

    def test_config_loading(self):
        """Test configuration loading from YAML."""
        self._save_config(self.basic_config)

        config = Config.load(self.config_file)

        self.assertTrue(config.multi_agent.enabled)
        self.assertIn("information", config.multi_agent.agents)
        self.assertEqual(config.tts.provider, "openai")
        self.assertEqual(config.stt.provider, "openai")

    def test_config_validation(self):
        """Test configuration validation."""
        self._save_config(self.basic_config)
        config = Config.load(self.config_file)

        # Should pass validation with basic config
        issues = config.validate_multi_agent_config()
        self.assertEqual(len(issues), 0)

        # Test with invalid configuration
        invalid_config = self.basic_config.copy()
        invalid_config["multi_agent"]["agents"] = {}  # No agents defined
        self._save_config(invalid_config)

        config = Config.load(self.config_file)
        issues = config.validate_multi_agent_config()
        self.assertGreater(len(issues), 0)

    def test_health_checks(self):
        """Test configuration health checks."""
        self._save_config(self.basic_config)
        config = Config.load(self.config_file)

        health_results = config.run_health_checks()

        self.assertIn("overall_health", health_results)
        self.assertIn("checks", health_results)
        self.assertIn("timestamp", health_results)

        # Should have multiple health check categories
        checks = health_results["checks"]
        self.assertIn("multi_agent_validation", checks)
        self.assertIn("configuration_completeness", checks)
        self.assertIn("tool_availability", checks)
        self.assertIn("resource_limits", checks)

    def test_migration_functionality(self):
        """Test single-agent to multi-agent migration."""
        # Create single-agent configuration
        single_agent_config = {
            "multi_agent": {"enabled": False},
            "tts": {"provider": "openai", "voice": "nova"},
            "stt": {"provider": "openai"},
        }
        self._save_config(single_agent_config)

        config = Config.load(self.config_file)
        self.assertFalse(config.multi_agent.enabled)

        # Perform migration
        config.migrate_from_single_agent()

        # Verify migration results
        self.assertTrue(config.multi_agent.enabled)
        self.assertGreater(len(config.multi_agent.agents), 0)
        self.assertIn("general", config.multi_agent.agents)

    def test_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        self._save_config(self.basic_config)
        config = Config.load(self.config_file)

        # Create backup
        backup_data = config.create_backup()

        self.assertIn("timestamp", backup_data)
        self.assertIn("config_data", backup_data)
        self.assertIn("metadata", backup_data)

        # Modify configuration
        config.multi_agent.enabled = False

        # Restore from backup
        success = config.restore_from_backup(backup_data)
        self.assertTrue(success)
        self.assertTrue(config.multi_agent.enabled)

    def test_templates_system(self):
        """Test configuration templates."""
        # Test template listing
        available_templates = self.templates.list_templates()
        self.assertIn("basic", available_templates)
        self.assertIn("advanced", available_templates)
        self.assertIn("production", available_templates)

        # Test template retrieval
        basic_template = self.templates.get_template("basic")
        self.assertIn("multi_agent", basic_template)
        self.assertTrue(basic_template["multi_agent"]["enabled"])

        # Test template descriptions
        description = self.templates.get_template_description("basic")
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)

    def test_cli_manager_validation(self):
        """Test CLI manager validation command."""
        self._save_config(self.basic_config)

        cli_manager = CLIConfigManager(self.config_file)

        # Test validation with valid config
        result = cli_manager.validate_config(self.config_file, strict=False)
        self.assertEqual(result, 0)

        # Test validation with invalid config
        invalid_config = {"invalid": "structure"}
        self._save_config(invalid_config)

        result = cli_manager.validate_config(self.config_file, strict=True)
        self.assertEqual(result, 1)

    def test_cli_manager_health_checks(self):
        """Test CLI manager health check command."""
        self._save_config(self.basic_config)

        cli_manager = CLIConfigManager(self.config_file)
        result = cli_manager.run_health_checks(self.config_file, "json")

        # Should return 0 for healthy configuration
        self.assertEqual(result, 0)

    def test_cli_manager_templates(self):
        """Test CLI manager template operations."""
        cli_manager = CLIConfigManager(self.config_file)

        # Test template listing
        result = cli_manager.list_templates()
        self.assertEqual(result, 0)

        # Test template application
        result = cli_manager.apply_template("basic", self.config_file)
        self.assertEqual(result, 0)

        # Verify template was applied
        self.assertTrue(self.config_file.exists())
        config = Config.load(self.config_file)
        self.assertTrue(config.multi_agent.enabled)

    def test_cli_manager_migration(self):
        """Test CLI manager migration functionality."""
        # Create single-agent config
        single_agent_config = {
            "multi_agent": {"enabled": False},
            "tts": {"provider": "openai"},
        }
        self._save_config(single_agent_config)

        cli_manager = CLIConfigManager(self.config_file)
        result = cli_manager.migrate_single_to_multi(self.config_file, backup=True)

        self.assertEqual(result, 0)

        # Verify migration
        config = Config.load(self.config_file)
        self.assertTrue(config.multi_agent.enabled)

        # Verify backup was created
        backup_file = self.config_file.with_suffix(".backup.json")
        self.assertTrue(backup_file.exists())

    def test_cli_manager_backup_restore(self):
        """Test CLI manager backup and restore operations."""
        self._save_config(self.basic_config)

        cli_manager = CLIConfigManager(self.config_file)
        backup_file = self.temp_dir / "backup.json"

        # Test backup creation
        result = cli_manager.create_backup(self.config_file, backup_file)
        self.assertEqual(result, 0)
        self.assertTrue(backup_file.exists())

        # Modify configuration
        modified_config = self.basic_config.copy()
        modified_config["multi_agent"]["enabled"] = False
        self._save_config(modified_config)

        # Test restore
        result = cli_manager.restore_backup(backup_file, self.config_file)
        self.assertEqual(result, 0)

        # Verify restore
        config = Config.load(self.config_file)
        self.assertTrue(config.multi_agent.enabled)

    def test_cli_manager_set_get_values(self):
        """Test CLI manager set and get value operations."""
        self._save_config(self.basic_config)

        cli_manager = CLIConfigManager(self.config_file)

        # Test setting values
        result = cli_manager.set_config_values(
            ["multi_agent.enabled=false", "tts.voice=alloy"], self.config_file
        )
        self.assertEqual(result, 0)

        # Verify values were set
        config = Config.load(self.config_file)
        self.assertFalse(config.multi_agent.enabled)
        self.assertEqual(config.tts.voice, "alloy")

        # Test getting values
        result = cli_manager.get_config_values(
            ["multi_agent.enabled", "tts.voice"], self.config_file
        )
        self.assertEqual(result, 0)

    def test_agent_configurations(self):
        """Test specific agent configurations."""
        # Apply advanced template which has all agents
        advanced_config = self.templates.get_template("advanced")
        self._save_config(advanced_config)

        config = Config.load(self.config_file)

        # Verify all agents are configured
        agents = config.multi_agent.agents
        self.assertIn("information", agents)
        self.assertIn("productivity", agents)
        self.assertIn("utility", agents)
        self.assertIn("general", agents)
        self.assertIn("tool_specialist", agents)

        # Verify agent configurations
        info_agent = agents["information"]
        self.assertTrue(info_agent.enabled)
        self.assertIsNotNone(info_agent.model_config)
        self.assertGreater(len(info_agent.capabilities), 0)

    def test_routing_rules(self):
        """Test routing rule configurations."""
        advanced_config = self.templates.get_template("advanced")
        self._save_config(advanced_config)

        config = Config.load(self.config_file)

        # Verify routing rules exist
        routing = config.multi_agent.routing
        self.assertGreater(len(routing.rules), 0)

        # Check routing rule structure
        first_rule = routing.rules[0]
        self.assertIsNotNone(first_rule.patterns)
        self.assertIsNotNone(first_rule.agent)

    def test_workflow_orchestration(self):
        """Test workflow orchestration configuration."""
        advanced_config = self.templates.get_template("advanced")
        self._save_config(advanced_config)

        config = Config.load(self.config_file)

        # Verify workflow orchestration settings
        orchestration = config.multi_agent.workflow_orchestration
        self.assertIsNotNone(orchestration.max_parallel_tasks)
        self.assertIsNotNone(orchestration.task_timeout)
        self.assertIsNotNone(orchestration.retry_policy)

    def test_communication_settings(self):
        """Test inter-agent communication configuration."""
        advanced_config = self.templates.get_template("advanced")
        self._save_config(advanced_config)

        config = Config.load(self.config_file)

        # Verify communication settings
        communication = config.multi_agent.communication
        self.assertIsNotNone(communication.message_queue_size)
        self.assertIsNotNone(communication.heartbeat_interval)
        self.assertIsNotNone(communication.timeout_settings)

    def test_performance_tuning(self):
        """Test performance tuning configuration."""
        production_config = self.templates.get_template("production")
        self._save_config(production_config)

        config = Config.load(self.config_file)

        # Verify performance settings
        performance = config.multi_agent.performance_tuning
        self.assertIsNotNone(performance.load_balancing)
        self.assertIsNotNone(performance.resource_allocation)
        self.assertIsNotNone(performance.caching)

    def test_security_settings(self):
        """Test security configuration."""
        production_config = self.templates.get_template("production")
        self._save_config(production_config)

        config = Config.load(self.config_file)

        # Verify security settings
        security = config.multi_agent.security
        self.assertIsNotNone(security.rate_limiting)
        self.assertIsNotNone(security.audit_logging)
        self.assertIsNotNone(security.access_control)

    def test_monitoring_configuration(self):
        """Test monitoring and logging configuration."""
        production_config = self.templates.get_template("production")
        self._save_config(production_config)

        config = Config.load(self.config_file)

        # Verify monitoring settings
        monitoring = config.multi_agent.monitoring
        self.assertIsNotNone(monitoring.metrics_collection)
        self.assertIsNotNone(monitoring.health_checks)
        self.assertIsNotNone(monitoring.logging)


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration system."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "integration_config.yaml"
        self.templates = ConfigurationTemplates()

    def tearDown(self):
        """Clean up integration test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """Test complete configuration workflow."""
        cli_manager = CLIConfigManager(self.config_file)

        # 1. Apply production template
        result = cli_manager.apply_template("production", self.config_file)
        self.assertEqual(result, 0)

        # 2. Validate configuration
        result = cli_manager.validate_config(self.config_file)
        self.assertEqual(result, 0)

        # 3. Run health checks
        result = cli_manager.run_health_checks(self.config_file)
        self.assertEqual(result, 0)

        # 4. Create backup
        backup_file = self.temp_dir / "backup.json"
        result = cli_manager.create_backup(self.config_file, backup_file)
        self.assertEqual(result, 0)

        # 5. Modify configuration
        result = cli_manager.set_config_values(
            ["multi_agent.workflow_orchestration.max_parallel_tasks=10"],
            self.config_file,
        )
        self.assertEqual(result, 0)

        # 6. Verify changes
        config = Config.load(self.config_file)
        self.assertEqual(
            config.multi_agent.workflow_orchestration.max_parallel_tasks, 10
        )

        # 7. Restore from backup
        result = cli_manager.restore_backup(backup_file, self.config_file)
        self.assertEqual(result, 0)

        # 8. Verify restoration
        config = Config.load(self.config_file)
        self.assertNotEqual(
            config.multi_agent.workflow_orchestration.max_parallel_tasks, 10
        )


def run_tests():
    """Run all configuration tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestConfigurationSystem))
    test_suite.addTest(unittest.makeSuite(TestConfigurationIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
