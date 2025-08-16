"""
CLI Configuration Manager for Voice Agent.

Provides command-line interface for configuration management,
including template application, validation, backup/restore,
and quick configuration changes.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from voice_agent.core.config import Config
from voice_agent.config.templates import ConfigurationTemplates


class CLIConfigManager:
    """Command-line interface for configuration management."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize CLI configuration manager."""
        self.config_path = config_path or Path("src/voice_agent/config/default.yaml")
        self.templates = ConfigurationTemplates()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for configuration CLI."""
        parser = argparse.ArgumentParser(
            prog="voice-agent-config",
            description="Voice Agent Configuration Management Tool",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Validate command
        validate_parser = subparsers.add_parser(
            "validate", help="Validate configuration file"
        )
        validate_parser.add_argument(
            "--config",
            "-c",
            type=Path,
            help="Path to configuration file (default: default.yaml)",
        )
        validate_parser.add_argument(
            "--strict",
            action="store_true",
            help="Use strict validation (fail on warnings)",
        )

        # Health check command
        health_parser = subparsers.add_parser(
            "health", help="Run configuration health checks"
        )
        health_parser.add_argument(
            "--config", "-c", type=Path, help="Path to configuration file"
        )
        health_parser.add_argument(
            "--format", choices=["json", "text"], default="text", help="Output format"
        )

        # Template commands
        template_parser = subparsers.add_parser("template", help="Template management")
        template_subparsers = template_parser.add_subparsers(dest="template_action")

        # List templates
        template_subparsers.add_parser("list", help="List available templates")

        # Apply template
        apply_template_parser = template_subparsers.add_parser(
            "apply", help="Apply configuration template"
        )
        apply_template_parser.add_argument(
            "template_name",
            choices=self.templates.list_templates(),
            help="Template to apply",
        )
        apply_template_parser.add_argument(
            "--output", "-o", type=Path, help="Output configuration file path"
        )
        apply_template_parser.add_argument(
            "--merge",
            action="store_true",
            help="Merge with existing configuration instead of replacing",
        )

        # Show template
        show_template_parser = template_subparsers.add_parser(
            "show", help="Show template configuration"
        )
        show_template_parser.add_argument(
            "template_name",
            choices=self.templates.list_templates(),
            help="Template to show",
        )
        show_template_parser.add_argument(
            "--format", choices=["json", "yaml"], default="yaml", help="Output format"
        )

        # Migration commands
        migrate_parser = subparsers.add_parser("migrate", help="Migration utilities")
        migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_action")

        # Single to multi-agent migration
        single_to_multi_parser = migrate_subparsers.add_parser(
            "single-to-multi",
            help="Migrate from single-agent to multi-agent configuration",
        )
        single_to_multi_parser.add_argument(
            "--config", "-c", type=Path, help="Configuration file to migrate"
        )
        single_to_multi_parser.add_argument(
            "--backup", action="store_true", help="Create backup before migration"
        )

        # Backup/Restore commands
        backup_parser = subparsers.add_parser(
            "backup", help="Create configuration backup"
        )
        backup_parser.add_argument(
            "--config", "-c", type=Path, help="Configuration file to backup"
        )
        backup_parser.add_argument("--output", "-o", type=Path, help="Backup file path")

        restore_parser = subparsers.add_parser(
            "restore", help="Restore configuration from backup"
        )
        restore_parser.add_argument(
            "backup_file", type=Path, help="Backup file to restore from"
        )
        restore_parser.add_argument(
            "--config", "-c", type=Path, help="Configuration file to restore to"
        )

        # Quick configuration commands
        set_parser = subparsers.add_parser("set", help="Set configuration values")
        set_parser.add_argument(
            "key_value",
            nargs="+",
            help="Key=value pairs to set (e.g., multi_agent.enabled=true)",
        )
        set_parser.add_argument(
            "--config", "-c", type=Path, help="Configuration file to modify"
        )

        get_parser = subparsers.add_parser("get", help="Get configuration values")
        get_parser.add_argument(
            "keys", nargs="+", help="Configuration keys to retrieve"
        )
        get_parser.add_argument(
            "--config", "-c", type=Path, help="Configuration file to read from"
        )

        return parser

    def validate_config(
        self, config_path: Optional[Path] = None, strict: bool = False
    ) -> int:
        """Validate configuration file."""
        config_file = config_path or self.config_path

        try:
            config = Config.load(config_file)
            issues = config.validate_multi_agent_config()

            if not issues:
                print(f"✅ Configuration {config_file} is valid")
                return 0

            print(f"⚠️  Configuration {config_file} has issues:")
            for issue in issues:
                print(f"  - {issue}")

            return 1 if strict else 0

        except Exception as e:
            print(f"❌ Failed to validate configuration: {e}")
            return 1

    def run_health_checks(
        self, config_path: Optional[Path] = None, format_type: str = "text"
    ) -> int:
        """Run configuration health checks."""
        config_file = config_path or self.config_path

        try:
            config = Config.load(config_file)
            health_results = config.run_health_checks()

            if format_type == "json":
                print(json.dumps(health_results, indent=2))
            else:
                self._print_health_results(health_results)

            return 0 if health_results["overall_health"] == "healthy" else 1

        except Exception as e:
            print(f"❌ Failed to run health checks: {e}")
            return 1

    def _print_health_results(self, health_results: Dict[str, Any]) -> None:
        """Print health check results in text format."""
        overall = health_results["overall_health"]

        if overall == "healthy":
            print("✅ Overall Health: HEALTHY")
        elif overall == "degraded":
            print("⚠️  Overall Health: DEGRADED")
        else:
            print("❌ Overall Health: UNHEALTHY")

        print(f"Timestamp: {health_results['timestamp']}")
        print("\nHealth Checks:")

        for check_name, check_result in health_results["checks"].items():
            status = check_result["status"]
            if status == "pass":
                print(f"  ✅ {check_name}: PASS")
            elif status == "warning":
                print(f"  ⚠️  {check_name}: WARNING")
            else:
                print(f"  ❌ {check_name}: FAIL")

            if check_result["issues"]:
                for issue in check_result["issues"]:
                    print(f"    - {issue}")

    def list_templates(self) -> int:
        """List available configuration templates."""
        templates = self.templates.list_templates()

        print("Available Configuration Templates:")
        for template in templates:
            description = self.templates.get_template_description(template)
            print(f"  {template}: {description}")

        return 0

    def apply_template(
        self,
        template_name: str,
        output_path: Optional[Path] = None,
        merge: bool = False,
    ) -> int:
        """Apply configuration template."""
        try:
            template_config = self.templates.get_template(template_name)
            target_path = output_path or self.config_path

            if merge and target_path.exists():
                # Load existing config and merge
                existing_config = Config.load(target_path)
                # This would need deep merge logic - simplified for now
                existing_dict = existing_config.model_dump()
                self._deep_merge(existing_dict, template_config)
                template_config = existing_dict

            # Save template to YAML file
            with open(target_path, "w") as f:
                yaml.dump(template_config, f, default_flow_style=False)

            action = "merged with" if merge else "applied to"
            print(f"✅ Template '{template_name}' {action} {target_path}")
            return 0

        except Exception as e:
            print(f"❌ Failed to apply template: {e}")
            return 1

    def show_template(self, template_name: str, format_type: str = "yaml") -> int:
        """Show template configuration."""
        try:
            template_config = self.templates.get_template(template_name)

            if format_type == "json":
                print(json.dumps(template_config, indent=2))
            else:
                print(yaml.dump(template_config, default_flow_style=False))

            return 0

        except Exception as e:
            print(f"❌ Failed to show template: {e}")
            return 1

    def migrate_single_to_multi(
        self, config_path: Optional[Path] = None, backup: bool = False
    ) -> int:
        """Migrate from single-agent to multi-agent configuration."""
        config_file = config_path or self.config_path

        try:
            config = Config.load(config_file)

            if config.multi_agent.enabled:
                print("ℹ️  Configuration already uses multi-agent setup")
                return 0

            if backup:
                backup_data = config.create_backup()
                backup_path = config_file.with_suffix(".backup.json")
                with open(backup_path, "w") as f:
                    json.dump(backup_data, f, indent=2)
                print(f"📦 Backup created: {backup_path}")

            config.migrate_from_single_agent()
            config.save(config_file)

            print(f"✅ Configuration migrated to multi-agent: {config_file}")
            return 0

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return 1

    def create_backup(
        self, config_path: Optional[Path] = None, output_path: Optional[Path] = None
    ) -> int:
        """Create configuration backup."""
        config_file = config_path or self.config_path

        try:
            config = Config.load(config_file)
            backup_data = config.create_backup()

            if output_path:
                backup_file = output_path
            else:
                backup_file = config_file.with_suffix(".backup.json")

            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            print(f"✅ Backup created: {backup_file}")
            return 0

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return 1

    def restore_backup(
        self, backup_file: Path, config_path: Optional[Path] = None
    ) -> int:
        """Restore configuration from backup."""
        config_file = config_path or self.config_path

        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)

            config = Config.load(config_file)

            if config.restore_from_backup(backup_data):
                config.save(config_file)
                print(f"✅ Configuration restored from {backup_file}")
                return 0
            else:
                print("❌ Failed to restore configuration")
                return 1

        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return 1

    def set_config_values(
        self, key_values: List[str], config_path: Optional[Path] = None
    ) -> int:
        """Set configuration values."""
        config_file = config_path or self.config_path

        try:
            config = Config.load(config_file)

            for kv in key_values:
                if "=" not in kv:
                    print(f"❌ Invalid key=value format: {kv}")
                    return 1

                key, value = kv.split("=", 1)
                self._set_nested_value(config, key, value)

            config.save(config_file)
            print(f"✅ Configuration updated: {config_file}")
            return 0

        except Exception as e:
            print(f"❌ Failed to set configuration values: {e}")
            return 1

    def get_config_values(
        self, keys: List[str], config_path: Optional[Path] = None
    ) -> int:
        """Get configuration values."""
        config_file = config_path or self.config_path

        try:
            config = Config.load(config_file)

            for key in keys:
                value = self._get_nested_value(config, key)
                print(f"{key}={value}")

            return 0

        except Exception as e:
            print(f"❌ Failed to get configuration values: {e}")
            return 1

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target dictionary."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _set_nested_value(self, config: Config, key: str, value: str) -> None:
        """Set nested configuration value."""
        parts = key.split(".")
        obj = config

        for part in parts[:-1]:
            obj = getattr(obj, part)

        # Convert string value to appropriate type
        final_value = self._convert_value(value)
        setattr(obj, parts[-1], final_value)

    def _get_nested_value(self, config: Config, key: str) -> Any:
        """Get nested configuration value."""
        parts = key.split(".")
        obj = config

        for part in parts:
            obj = getattr(obj, part)

        return obj

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run CLI configuration manager."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return 1

        try:
            # Route to appropriate command handler
            if parsed_args.command == "validate":
                return self.validate_config(parsed_args.config, parsed_args.strict)

            elif parsed_args.command == "health":
                return self.run_health_checks(parsed_args.config, parsed_args.format)

            elif parsed_args.command == "template":
                if parsed_args.template_action == "list":
                    return self.list_templates()
                elif parsed_args.template_action == "apply":
                    return self.apply_template(
                        parsed_args.template_name, parsed_args.output, parsed_args.merge
                    )
                elif parsed_args.template_action == "show":
                    return self.show_template(
                        parsed_args.template_name, parsed_args.format
                    )

            elif parsed_args.command == "migrate":
                if parsed_args.migrate_action == "single-to-multi":
                    return self.migrate_single_to_multi(
                        parsed_args.config, parsed_args.backup
                    )

            elif parsed_args.command == "backup":
                return self.create_backup(parsed_args.config, parsed_args.output)

            elif parsed_args.command == "restore":
                return self.restore_backup(parsed_args.backup_file, parsed_args.config)

            elif parsed_args.command == "set":
                return self.set_config_values(parsed_args.key_value, parsed_args.config)

            elif parsed_args.command == "get":
                return self.get_config_values(parsed_args.keys, parsed_args.config)

            else:
                print(f"❌ Unknown command: {parsed_args.command}")
                return 1

        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1


def main():
    """Main CLI entry point."""
    manager = CLIConfigManager()
    return manager.run()


if __name__ == "__main__":
    sys.exit(main())
