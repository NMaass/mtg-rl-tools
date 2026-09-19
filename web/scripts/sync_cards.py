"""Build a static catalogue from Scryfall bulk JSON or gzipped JSONL."""
import argparse
import datetime
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.parse
import urllib.request

HEADERS = {'User-Agent': 'PriorityReplay/0.1 (https://github.com/NMaass/mtg-rl-tools)', 'Accept': 'application/json;q=0.9,*/*;q=0.8'}
EXAMPLES = {'Mountain', 'Island', 'Monastery Swiftspear', 'Delver of Secrets', 'Lightning Bolt', 'Lava Spike', 'Play with Fire', 'Goblin Guide', 'Ponder'}


def get(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ('api.scryfall.com', 'data.scryfall.io', 'cards.scryfall.io'):
        raise ValueError('Unexpected card-data host.')
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)


def read_cards(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == '[':
            yield from json.load(handle)
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def build(cards, out, images=True):
    arena, names, art = {}, {}, {}
    for c in cards:
        if not isinstance(c, dict) or c.get('lang') != 'en':
            continue
        face = (c.get('card_faces') or [c])[0]
        name = c.get('name', '')
        entry = {'name': name, 'manaCost': c.get('mana_cost', face.get('mana_cost', '')),
                 'oracleText': c.get('oracle_text', face.get('oracle_text', '')),
                 'types': [c.get('type_line', face.get('type_line', ''))], 'art': c['id']}
        if c.get('arena_id') is not None:
            arena[str(c['arena_id'])] = entry
        if name not in names or 'paper' in (c.get('games') or []):
            names[name] = entry
        image = (c.get('image_uris') or face.get('image_uris') or {}).get('art_crop')
        if name in EXAMPLES and image:
            art[name] = (c['id'], image)
    if not arena or not names:
        raise ValueError('Bulk data contained no usable Arena identities. Catalogue not published.')
    out.mkdir(parents=True, exist_ok=True)
    payload = {'version': 1, 'updated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source': 'Scryfall default_cards', 'arena': arena, 'names': names}
    serialized = json.dumps(payload, separators=(',', ':'), ensure_ascii=True)
    if len(serialized.encode()) > 24 * 1024 * 1024:
        raise ValueError('Catalogue exceeds 24 MiB; partition before publishing.')
    (out / 'catalog.json').write_text(serialized)
    if images:
        target = out / 'example-art'
        target.mkdir(exist_ok=True)
        for name, (identity, url) in art.items():
            with get(url) as response:
                data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise ValueError('Example image exceeds the image budget.')
            (target / (identity + '.jpg')).write_bytes(data)
            time.sleep(.15)
        (out / 'example-art.json').write_text(json.dumps({name: identity for name, (identity, _) in art.items()}))
    print('Catalogue: %d Arena IDs, %d names, %.1f MiB' % (len(arena), len(names), len(serialized) / 1024 / 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bulk', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'public')
    parser.add_argument('--no-images', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        source = args.bulk
        if source is None:
            with get('https://api.scryfall.com/bulk-data') as response:
                listing = json.load(response)
            metadata = next((entry for entry in listing.get('data', []) if entry.get('type') == 'default_cards'), None)
            uri = (metadata or {}).get('jsonl_download_uri') or (metadata or {}).get('download_uri')
            if not uri:
                keys = sorted((metadata or listing).keys())
                raise ValueError('No supported Scryfall bulk URI. Descriptor fields: ' + ', '.join(keys))
            print('Bulk format: ' + ('JSONL gzip' if '.jsonl.gz' in uri else 'JSON'))
            source = Path(tmp) / ('default-cards.jsonl.gz' if '.gz' in urllib.parse.urlparse(uri).path else 'default-cards.json')
            with get(uri) as response, source.open('wb') as handle:
                copied = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > 1024 * 1024 * 1024:
                        raise ValueError('Bulk download exceeds 1 GiB.')
                    handle.write(chunk)
        build(read_cards(source), args.out, not args.no_images)


if __name__ == '__main__':
    main()
