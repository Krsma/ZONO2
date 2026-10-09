"""Populate a persistent workspace once without replacing student edits."""
import os
from pathlib import Path
import shutil
import sys

for source in Path('/opt/pzr/notebooks').glob('*.ipynb'):
    destination = Path('/workspace') / source.name
    if not destination.exists():
        shutil.copyfile(source, destination)
os.execvp(sys.argv[1], sys.argv[1:])
