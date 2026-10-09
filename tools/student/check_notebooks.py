"""Execute onboarding notebooks in a fresh writable directory."""
import argparse
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--notebooks', type=Path, default=Path(__file__).resolve().parents[2] / 'notebooks')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix='pzr-notebooks-') as temporary:
    root = args.output or Path(temporary)
    root.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.notebooks.glob('0*.ipynb')):
        print(f'Executing {path.name}', flush=True)
        book = nbformat.read(path, as_version=4)
        NotebookClient(book, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
        if args.output:
            nbformat.write(book, root / path.name)
        print(f'Passed {path.name}', flush=True)
