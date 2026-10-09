"""Vendor the pinned native source for credential-free, offline Cargo builds."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BINDING = '01c92a2bfac58755e3b832bb0094816f3f36e1d1'
INTERPRETER = '2724b05ae6c62ed0df14f1401ed8db89472725a6'
source = ROOT / 'rlolapythonbinding'
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if revision != BINDING:
    raise SystemExit(f'Unexpected binding revision: {revision}')
if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
    raise SystemExit('Binding checkout must be clean for a provenance-checked build')
lock = (source / 'Cargo.lock').read_text()
if lock.count('rtlola-interpreter.git?branch=slack-vars#' + INTERPRETER) != 2:
    raise SystemExit('Unexpected interpreter revision in Cargo.lock')
output = ROOT / 'build/student-native'
output.mkdir(parents=True, exist_ok=True)
for name in ('Cargo.toml', 'Cargo.lock', 'pyproject.toml'):
    if (source / name).exists():
        shutil.copyfile(source / name, output / name)
shutil.copytree(source / 'src', output / 'src', dirs_exist_ok=True)
config = subprocess.check_output([
    'cargo', 'vendor', '--offline', '--locked', '--manifest-path', str(source / 'Cargo.toml'),
    str(output / 'vendor'),
], text=True)
# Cargo config directory is fixed inside the builder, never a host absolute path.
config = config.replace(str(output / 'vendor'), '/native/vendor')
(output / '.cargo').mkdir(exist_ok=True)
(output / '.cargo/config.toml').write_text(config)
(output / 'provenance.json').write_text(json.dumps({
    'binding_revision': BINDING, 'interpreter_revision': INTERPRETER, 'build_profile': 'release',
}, indent=2) + '\n')
print(output)
