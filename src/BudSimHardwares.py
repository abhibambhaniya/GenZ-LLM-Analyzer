from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from Systems.system_configs import system_configs

from .utils import sqlite_utils


class Hardware(BaseModel):
    """Pydantic model mirroring a single hardware configuration."""

    Flops: int
    Memory_size: int
    Memory_BW: int
    ICN: int
    ICN_LL: Optional[float] = None
    real_values: bool = True


class BudHardwares:
    """Utility class to manage Bud simulator hardware presets with SQLite persistence."""

    def __init__(
        self,
        hardwares: Optional[Dict[str, Dict[str, Any]]] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.conn = sqlite_utils.get_connection(db_path)

        existing = sqlite_utils.fetch_all(self.conn)
        if not existing:
            if hardwares is None:
                hardwares = system_configs
            for name, cfg in hardwares.items():
                sqlite_utils.upsert(self.conn, name, cfg)
        elif hardwares:
            for name, cfg in hardwares.items():
                sqlite_utils.upsert(self.conn, name, cfg)

        self._hardwares: Dict[str, Hardware] = {
            name: Hardware(**cfg)
            for name, cfg in sqlite_utils.fetch_all(self.conn).items()
        }

    def list_hardwares(self) -> List[str]:
        """Return a list of supported hardware names."""
        return list(self._hardwares.keys())

    def get_hardware(self, name: str) -> Hardware:
        """Retrieve a hardware configuration by name."""
        return self._hardwares[name]

    def add_hardware(self, name: str, config: Dict[str, Any]) -> None:
        """Add a new hardware configuration."""
        hw = Hardware(**config)
        self._hardwares[name] = hw
        sqlite_utils.upsert(self.conn, name, hw.dict())

    def update_hardware(self, name: str, config: Dict[str, Any]) -> None:
        """Update an existing hardware configuration."""
        if name not in self._hardwares:
            raise KeyError(f"Hardware {name} not found")
        self._hardwares[name] = self._hardwares[name].copy(update=config)
        sqlite_utils.upsert(self.conn, name, self._hardwares[name].dict())

    def delete_hardware(self, name: str) -> None:
        """Delete a hardware configuration."""
        if name not in self._hardwares:
            raise KeyError(f"Hardware {name} not found")
        del self._hardwares[name]
        sqlite_utils.delete(self.conn, name)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return all hardware configurations as a dictionary."""
        return {name: hw.dict() for name, hw in self._hardwares.items()}

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
