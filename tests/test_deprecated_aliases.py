"""Old names (System, *_moddeling, model_profilling) must still work, with a DeprecationWarning."""
import pytest

import GenZ
from GenZ import NPU, prefill_modeling


def test_old_names_resolve_to_new_objects():
    with pytest.warns(DeprecationWarning):
        from GenZ import System
    assert System is NPU

    with pytest.warns(DeprecationWarning):
        from GenZ.system import System as OldPathSystem
    assert OldPathSystem is NPU

    with pytest.warns(DeprecationWarning):
        assert GenZ.prefill_moddeling is prefill_modeling

    with pytest.raises(AttributeError):
        GenZ.NoSuchName


def test_old_kwarg_is_renamed():
    kw = dict(model="gpt-2", batch_size=1, input_tokens=128, system_name="A100_40GB_GPU")
    with pytest.warns(DeprecationWarning):
        old = prefill_modeling(**kw, model_profilling=True)
    new = prefill_modeling(**kw, model_profiling=True)
    assert old[1].equals(new[1])  # (model_df, summary_table)
