"""Run with python -m pzr.student.doctor to check a student environment."""
from importlib.metadata import version
import json
import platform

from pzr.student import load_task


def main() -> None:
    task = load_task()
    report = {'python': platform.python_version(), 'architecture': platform.machine(),
              'starter_samples': task.dataset.num_samples,
              'packages': {name: version(name) for name in ('torch', 'numpy', 'gymnasium', 'jupyterlab')}}
    from pzr.rtlola.binding import BINDING_REVISION, INTERPRETER_REVISION, require_binding
    require_binding()
    report.update(binding_revision=BINDING_REVISION, interpreter_revision=INTERPRETER_REVISION)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
