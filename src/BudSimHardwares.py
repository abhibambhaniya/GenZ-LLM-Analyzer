from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from Systems.system_configs import system_configs

class Hardware(BaseModel):
    """Pydantic model mirroring a single hardware configuration."""
    Flops: int
    Memory_size: int
    Memory_BW: int
    ICN: int
    ICN_LL: Optional[float] = None
    real_values: bool = True

class BudHardwares:
    """Utility class to manage Bud simulator hardware presets."""

    def __init__(self, hardwares: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        if hardwares is None:
            hardwares = system_configs
        self._hardwares: Dict[str, Hardware] = {
            name: Hardware(**cfg) for name, cfg in hardwares.items()
        }

    def list_hardwares(self) -> List[str]:
        """Return a list of supported hardware names."""
        return list(self._hardwares.keys())

    def get_hardware(self, name: str) -> Hardware:
        """Retrieve a hardware configuration by name."""
        return self._hardwares[name]

    def add_hardware(self, name: str, config: Dict[str, Any]) -> None:
        """Add a new hardware configuration."""
        self._hardwares[name] = Hardware(**config)

    def update_hardware(self, name: str, config: Dict[str, Any]) -> None:
        """Update an existing hardware configuration."""
        if name not in self._hardwares:
            raise KeyError(f"Hardware {name} not found")
        self._hardwares[name] = self._hardwares[name].copy(update=config)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return all hardware configurations as a dictionary."""
        return {name: hw.dict() for name, hw in self._hardwares.items()}
