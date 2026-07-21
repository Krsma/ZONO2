"""Import guards for the optional RTLola Python binding."""

from __future__ import annotations

from importlib import import_module
from typing import Any

BINDING_REVISION = "01c92a2bfac58755e3b832bb0094816f3f36e1d1"
INTERPRETER_REVISION = "2724b05ae6c62ed0df14f1401ed8db89472725a6"
BINDING_BUILD_PROFILE = "release"
PROVENANCE_MODULE = "rlola_python_binding_pzr_provenance"


class RtlolaBindingUnavailable(RuntimeError):
    """Raised when the optional RTLola binding has not been installed."""


class RtlolaBindingMismatch(RtlolaBindingUnavailable):
    """Raised when the installed binding does not match experiment provenance."""


def require_binding() -> tuple[type[Any], type[Any], type[Any]]:
    """Return binding classes or raise a setup-oriented error."""
    try:
        import rlola_python_binding as binding
    except ImportError as exc:
        raise RtlolaBindingUnavailable(
            "rlola_python_binding is not importable. Run "
            "`tools/setup_rtlola_binding.sh` or build the pinned "
            "`rlolapythonbinding` submodule with "
            "maturin in the active Python environment."
        ) from exc
    actual_binding, actual_interpreter, actual_profile = _binding_metadata(binding)
    if (
        actual_binding != BINDING_REVISION
        or actual_interpreter != INTERPRETER_REVISION
        or actual_profile != BINDING_BUILD_PROFILE
    ):
        raise RtlolaBindingMismatch(
            "installed rlola_python_binding does not match the required release "
            f"build (binding={actual_binding!r}, interpreter={actual_interpreter!r}, "
            f"profile={actual_profile!r}; expected binding={BINDING_REVISION!r}, "
            f"interpreter={INTERPRETER_REVISION!r}, "
            f"profile={BINDING_BUILD_PROFILE!r}). "
            "Run `tools/setup_rtlola_binding.sh`."
        )
    return binding.EvaluatorState, binding.RLolaMonitor, binding.ZonotopeConfig


def _binding_metadata(binding: Any) -> tuple[str | None, str | None, str | None]:
    actual_interpreter = getattr(binding, "INTERPRETER_REVISION", None)
    actual_profile = getattr(binding, "BUILD_PROFILE", None)
    actual_binding = None
    try:
        provenance = import_module(PROVENANCE_MODULE)
    except ImportError:
        provenance = None
    if provenance is not None:
        actual_binding = getattr(provenance, "BINDING_REVISION", None)
        actual_interpreter = (
            actual_interpreter
            or getattr(provenance, "INTERPRETER_REVISION", None)
        )
        actual_profile = (
            actual_profile
            or getattr(provenance, "BINDING_BUILD_PROFILE", None)
        )
    return (
        actual_binding,
        actual_interpreter,
        actual_profile,
    )
