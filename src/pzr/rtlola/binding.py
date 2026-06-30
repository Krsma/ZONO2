"""Import guards for the optional RTLola Python binding."""

from __future__ import annotations

from typing import Any

BINDING_REVISION = "ca5976da9e105e48153b58e70f1f4d8c7aaa4cf6"


class RtlolaBindingUnavailable(RuntimeError):
    """Raised when the optional RTLola binding has not been installed."""


def require_binding() -> tuple[type[Any], type[Any], type[Any]]:
    """Return binding classes or raise a setup-oriented error."""
    try:
        from rlola_python_binding import EvaluatorState, RLolaMonitor, ZonotopeConfig
    except ImportError as exc:
        raise RtlolaBindingUnavailable(
            "rlola_python_binding is not importable. Run "
            "`tools/setup_rtlola_binding.sh` or build "
            "`vendor/rlola-python-binding` / `rlolapythonbinding` with "
            "maturin in the active Python environment."
        ) from exc
    return EvaluatorState, RLolaMonitor, ZonotopeConfig
