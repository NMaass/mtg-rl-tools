"""Use the existing Arena regression fixture on both Python and Wasm."""
from pathlib import Path
import sys
import json
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'python'),str(root/'python/tests')]
from test_arena_mirror import SessionAutoOpenTest
from magic_cabt.browser_export import parse_export
text=SessionAutoOpenTest()._log_text()
out=root/'web/public/fixtures'
out.mkdir(parents=True,exist_ok=True)
(out/'arena.log').write_text(text)
(out/'arena.expected.json').write_text(parse_export(text,'arena'))
