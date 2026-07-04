"""Import guards for the optional RTLola Python binding."""

from __future__ import annotations

from typing import Any

BINDING_REVISION = "abe3dab33d0c4aa504db0af63901b66ecafb7f71"
INTERPRETER_REVISION = "a143dd6a1500d54c1eabe9e83e5b54271734d6b2"
BINDING_BUILD_PROFILE = "release"


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
    actual_interpreter = getattr(binding, "INTERPRETER_REVISION", None)
    actual_profile = getattr(binding, "BUILD_PROFILE", None)
    if actual_interpreter != INTERPRETER_REVISION or actual_profile != BINDING_BUILD_PROFILE:
        raise RtlolaBindingMismatch(
            "installed rlola_python_binding does not match the required release "
            f"build (interpreter={actual_interpreter!r}, profile={actual_profile!r}; "
            f"expected interpreter={INTERPRETER_REVISION!r}, "
            f"profile={BINDING_BUILD_PROFILE!r}). "
            "Run `tools/setup_rtlola_binding.sh`."
        )
    return binding.EvaluatorState, binding.RLolaMonitor, binding.ZonotopeConfig
