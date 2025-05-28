import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Systems.bud_hardwares import BudHardwares


def test_hardware_persistence_db(tmp_path):
    db = tmp_path / "hw.db"
    hw = BudHardwares(db_path=str(db))
    hw.add_hardware("TEST_HW", {"Flops": 1, "Memory_size": 2, "Memory_BW": 3, "ICN": 4})

    hw2 = BudHardwares(db_path=str(db))
    assert "TEST_HW" in hw2.list_hardwares()
    assert hw2.get_hardware("TEST_HW").Flops == 1


def test_update_persistence_db(tmp_path):
    db = tmp_path / "hw.db"
    hw = BudHardwares(db_path=str(db))
    hw.add_hardware("TEST_HW", {"Flops": 1, "Memory_size": 2, "Memory_BW": 3, "ICN": 4})
    hw.update_hardware(
        "TEST_HW", {"Flops": 5, "Memory_size": 2, "Memory_BW": 3, "ICN": 4}
    )

    hw2 = BudHardwares(db_path=str(db))
    assert hw2.get_hardware("TEST_HW").Flops == 5
