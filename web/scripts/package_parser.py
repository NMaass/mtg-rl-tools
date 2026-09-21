"""Package the original parser sources for same-origin Pyodide execution."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parents[2]
out = root / 'web/public/parser'
out.mkdir(parents=True, exist_ok=True)
files = {}
paths = ['arena_log.py', 'arena_mirror/options.py', 'arena_mirror/tracker.py',
         'mtgo_video/parse.py', 'mtgo_video/state.py', 'mtgo_video/catalog.py',
         'mtgo_video/ocr.py', 'browser_export.py']
for relative in paths:
    source = root / 'python/magic_cabt' / relative
    if not source.is_file():
        raise SystemExit('Missing required parser source: ' + str(source))
    files['/app/magic_cabt/' + relative] = source.read_text()
for package in ('magic_cabt', 'magic_cabt/arena_mirror', 'magic_cabt/mtgo_video'):
    files['/app/' + package + '/__init__.py'] = ''
(out / 'sources.json').write_text(json.dumps(files))
(out / 'manifest.json').write_text(json.dumps({k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()}, indent=2))
runtime = root / 'web/node_modules/pyodide'
for name in ('pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json'):
    source = runtime / name
    if not source.is_file():
        raise SystemExit('Missing Pyodide runtime file: ' + str(source))
    shutil.copyfile(source, out / name)
