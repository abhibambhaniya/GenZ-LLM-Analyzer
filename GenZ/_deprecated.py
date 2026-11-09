"""Deprecated names, kept as aliases for one release.

The ICCAD'26 paper calls the hardware back-end an NPU (was ``System``), and the
misspelled ``*_moddeling`` / ``*_parallization`` / ``model_profilling`` names
are now spelled correctly. Old names still work but emit DeprecationWarning.
"""
import functools
import importlib
import sys
import warnings

MODULES = {"GenZ.system": "GenZ.npu"}

NAMES = {
    "System": "NPU",
    "ModdelingOutput": "ModelingOutput",
    "prefill_moddeling": "prefill_modeling",
    "decode_moddeling": "decode_modeling",
    "chunked_moddeling": "chunked_modeling",
    "get_various_parallization": "get_various_parallelization",
    "get_best_parallization_strategy": "get_best_parallelization_strategy",
}

KWARGS = {"model_profilling": "model_profiling"}

# Modules that define or re-export a renamed name.
_HOSTS = (
    "GenZ", "GenZ.npu", "GenZ.LLM_inference", "GenZ.LLM_inference.utils",
    "GenZ.LLM_inference.llm_prefill", "GenZ.LLM_inference.llm_decode",
    "GenZ.LLM_inference.llm_chunked", "GenZ.LLM_inference.llm_spec_decode",
    "GenZ.LLM_inference.best_parallelization",
)

# Functions that took the misspelled keyword argument.
_KWARG_FUNCS = (
    ("GenZ.LLM_inference.llm_prefill", "prefill_modeling"),
    ("GenZ.LLM_inference.llm_decode", "decode_modeling"),
    ("GenZ.LLM_inference.llm_chunked", "chunked_modeling"),
    ("GenZ.LLM_inference.llm_spec_decode", "spec_prefill_modeling"),
    ("GenZ.LLM_inference.llm_spec_decode", "spec_decode_modeling"),
)


def _warn(old: str, new: str) -> None:
    warnings.warn(f"'{old}' is deprecated; use '{new}'", DeprecationWarning, stacklevel=3)


def _module_getattr(module):
    def __getattr__(name):
        new = NAMES.get(name)
        if new is not None and new in module.__dict__:
            _warn(name, new)
            return module.__dict__[new]
        raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")
    return __getattr__


def _accept_old_kwargs(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for old, new in KWARGS.items():
            if old in kwargs:
                _warn(old, new)
                kwargs[new] = kwargs.pop(old)
        return fn(*args, **kwargs)
    return wrapper


def install() -> None:
    for old, new in MODULES.items():
        sys.modules.setdefault(old, importlib.import_module(new))
    for name in _HOSTS:
        module = importlib.import_module(name)
        module.__getattr__ = _module_getattr(module)

    # Rebind the wrapped function in every module that already imported it.
    for module_name, func_name in _KWARG_FUNCS:
        original = getattr(importlib.import_module(module_name), func_name)
        wrapped = _accept_old_kwargs(original)
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("GenZ") and module.__dict__.get(func_name) is original:
                setattr(module, func_name, wrapped)
