from Systems.bud_hardwares import BudHardwares


def test_hardware_persistence_json(tmp_path):
    file_path = tmp_path / "hw.json"
    hw = BudHardwares(config_file=str(file_path))
    hw.add_hardware("TEST_HW", {"Flops": 1, "Memory_size": 2, "Memory_BW": 3, "ICN": 4})

    hw2 = BudHardwares(config_file=str(file_path))
    assert "TEST_HW" in hw2
    assert hw2["TEST_HW"]["Flops"] == 1


def test_update_persistence_json(tmp_path):
    file_path = tmp_path / "hw.json"
    hw = BudHardwares(config_file=str(file_path))
    hw.add_hardware("TEST_HW", {"Flops": 1, "Memory_size": 2, "Memory_BW": 3, "ICN": 4})
    hw.update_hardware("TEST_HW", {"Flops": 5, "Memory_size": 2, "Memory_BW": 3, "ICN": 4})

    hw2 = BudHardwares(config_file=str(file_path))
    assert hw2["TEST_HW"]["Flops"] == 5
