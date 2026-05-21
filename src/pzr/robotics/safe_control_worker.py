"""Sidecar worker placeholder for safe-control-gym IROS integration."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-control-gym-root", required=True)
    parser.parse_args()
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("command")
        if command == "close":
            _write({"ok": True})
            return 0
        _write(
            {
                "ok": False,
                "error": (
                    "safe-control-gym sidecar protocol is installed, but the beta IROS "
                    "task binding still needs a concrete environment factory"
                ),
            }
        )
    return 0


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
