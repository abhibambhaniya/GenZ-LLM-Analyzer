import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.BudSimHardwares import BudHardwares, Hardware
from Systems.system_configs import system_configs


def test_list_contains_default_hardware(tmp_path):
    manager = BudHardwares(db_path=str(tmp_path / "hw.db"))
    for name in system_configs.keys():
        assert name in manager.list_hardwares()


def test_add_and_get_hardware(tmp_path):
    manager = BudHardwares(db_path=str(tmp_path / "hw.db"))
    new_cfg = {"Flops": 100, "Memory_size": 64, "Memory_BW": 2000, "ICN": 100}
    manager.add_hardware("TEST_HW", new_cfg)
    hw = manager.get_hardware("TEST_HW")
    assert isinstance(hw, Hardware)
    assert hw.Flops == 100

    # ensure persistence
    manager2 = BudHardwares(db_path=str(tmp_path / "hw.db"))
    assert manager2.get_hardware("TEST_HW").Flops == 100


def test_update_hardware(tmp_path):
    manager = BudHardwares(db_path=str(tmp_path / "hw.db"))
    name = manager.list_hardwares()[0]
    manager.update_hardware(name, {"Flops": 999})
    assert manager.get_hardware(name).Flops == 999

    manager2 = BudHardwares(db_path=str(tmp_path / "hw.db"))
    assert manager2.get_hardware(name).Flops == 999


def test_delete_hardware(tmp_path):
    manager = BudHardwares(db_path=str(tmp_path / "hw.db"))
    manager.add_hardware(
        "DEL", {"Flops": 1, "Memory_size": 2, "Memory_BW": 3, "ICN": 4}
    )
    manager.delete_hardware("DEL")
    with pytest.raises(KeyError):
        manager.get_hardware("DEL")
    manager2 = BudHardwares(db_path=str(tmp_path / "hw.db"))
    with pytest.raises(KeyError):
        manager2.get_hardware("DEL")
