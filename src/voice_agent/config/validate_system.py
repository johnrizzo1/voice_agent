#!/usr/bin/env python3
"""
Simple validation script to test the configuration system.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from voice_agent.core.config import Config
    from voice_agent.config.templates import ConfigurationTemplates

    print("✅ Successfully imported configuration modules")

    # Test template system
    templates = ConfigurationTemplates()
    available_templates = templates.list_templates()
    print(f"✅ Available templates: {', '.join(available_templates)}")

    # Test template loading
    basic_template = templates.get_template("basic")
    print("✅ Successfully loaded basic template")

    # Test template validation
    if "multi_agent" in basic_template and basic_template["multi_agent"]["enabled"]:
        print("✅ Basic template has multi-agent enabled")

    # Test advanced template
    advanced_template = templates.get_template("advanced")
    agents = advanced_template.get("multi_agent", {}).get("agents", {})
    print(f"✅ Advanced template has {len(agents)} agents configured")

    # Test production template
    production_template = templates.get_template("production")
    if "security" in production_template.get("multi_agent", {}):
        print("✅ Production template includes security configuration")

    print("\n🎉 Configuration system validation completed successfully!")
    print("\nSummary:")
    print(f"- Templates available: {len(available_templates)}")
    print(f"- Basic template: {'✅ Valid' if basic_template else '❌ Invalid'}")
    print(f"- Advanced template: {'✅ Valid' if advanced_template else '❌ Invalid'}")
    print(
        f"- Production template: {'✅ Valid' if production_template else '❌ Invalid'}"
    )

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Validation error: {e}")
    sys.exit(1)
