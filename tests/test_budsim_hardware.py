import pytest
from src.BudSimHardwares import BudHardwares, Hardware
from Systems.system_configs import system_configs


def test_list_contains_default_hardware():
    manager = BudHardwares()
    for name in system_configs.keys():
        assert name in manager.list_hardwares()


def test_add_and_get_hardware():
    manager = BudHardwares()
    new_cfg = {"Flops": 100, "Memory_size": 64, "Memory_BW": 2000, "ICN": 100}
    manager.add_hardware("TEST_HW", new_cfg)
    hw = manager.get_hardware("TEST_HW")
    assert isinstance(hw, Hardware)
    assert hw.Flops == 100


def test_update_hardware():
    manager = BudHardwares()
    name = manager.list_hardwares()[0]
    manager.update_hardware(name, {"Flops": 999})
    assert manager.get_hardware(name).Flops == 999
