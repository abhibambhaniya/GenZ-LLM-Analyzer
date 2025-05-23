from typing import Dict, Any, Optional
import json
import os

try:
    import yaml
except ImportError:  # pragma: no cover - YAML support optional
    yaml = None


class BudHardwares:
    """Manage hardware configurations with optional file persistence."""

    def __init__(self, config_file: Optional[str] = None, default_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.config_file = config_file
        self.hardwares: Dict[str, Dict[str, Any]] = {}
        if default_configs:
            self.hardwares.update(default_configs)
        if config_file:
            self._load_from_file(config_file)

    def _load_from_file(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            if path.endswith(".json"):
                data = json.load(f)
            elif path.endswith((".yaml", ".yml")):
                if yaml is None:
                    raise ImportError("PyYAML is required to load YAML files")
                data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file type for configuration file")
        if isinstance(data, dict):
            self.hardwares.update(data)

    def _save_to_file(self) -> None:
        if not self.config_file:
            return
        with open(self.config_file, "w") as f:
            if self.config_file.endswith(".json"):
                json.dump(self.hardwares, f, indent=2)
            elif self.config_file.endswith((".yaml", ".yml")):
                if yaml is None:
                    raise ImportError("PyYAML is required to save YAML files")
                yaml.safe_dump(self.hardwares, f)
            else:
                raise ValueError("Unsupported file type for configuration file")

    def add_hardware(self, name: str, config: Dict[str, Any]) -> None:
        self.hardwares[name] = config
        self._save_to_file()

    def update_hardware(self, name: str, config: Dict[str, Any]) -> None:
        self.hardwares[name] = config
        self._save_to_file()

    def get(self, name: str, default=None):
        return self.hardwares.get(name, default)

    def __contains__(self, name: str) -> bool:  # pragma: no cover - simple
        return name in self.hardwares

    def __getitem__(self, name: str) -> Dict[str, Any]:  # pragma: no cover - simple
        return self.hardwares[name]

    def items(self):  # pragma: no cover - simple
        return self.hardwares.items()

    def keys(self):  # pragma: no cover - simple
        return self.hardwares.keys()
