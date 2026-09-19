"""Build a compact static catalogue from Scryfall's bulk file, not per-card API scraping."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

HEADERS = {'User-Agent': 'PriorityReplay/0.1 (https://github.com/NMaass/mtg-rl-tools)', 'Accept': 'application/json;q=0.9,*/*;q=0.8'}
EXAMPLES = ['Mountain', 'Island', 'Monastery Swiftspear', 'Delver of Secrets', 'Lightning Bolt', 'Lava Spike', 'Play with Fire', 'Goblin Guide', 'Ponder']


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120)


def build(cards, out, images=True):
    arena, names = {}, {}
    art = {}
    for c in cards:
        if c.get('lang') != 'en':
            continue
        face = (c.get('card_faces') or [c])[0]
        name = c.get('name', '')
        entry = {'name': name, 'manaCost': c.get('mana_cost', face.get('mana_cost', '')),
                 'oracleText': c.get('oracle_text', face.get('oracle_text', '')),
                 'types': [c.get('type_line', face.get('type_line', ''))],
                 'art': c['id']}
        if c.get('arena_id') is not None:
            arena[str(c['arena_id'])] = entry
        if name not in names or c.get('games') and 'paper' in c['games']:
            names[name] = entry
        image = (c.get('image_uris') or face.get('image_uris') or {}).get('art_crop')
        if name in EXAMPLES and image:
            art[name] = (c['id'], image)
    out.mkdir(parents=True, exist_ok=True)
    payload = {'version': 1, 'updated': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source': 'Scryfall default_cards', 'arena': arena, 'names': names}
    catalogue = out / 'catalog.json'
    catalogue.write_text(json.dumps(payload, separators=(',', ':'), ensure_ascii=True))
    if catalogue.stat().st_size > 24 * 1024 * 1024:
        raise ValueError('Catalogue exceeds the static-asset size budget; partition before publishing.')
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
    print('Catalogue: %d Arena IDs, %d names, %.1f MiB' % (len(arena), len(names), catalogue.stat().st_size / 1024 / 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bulk', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'public')
    parser.add_argument('--no-images', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        source = args.bulk
        if source is None:
            with get('https://api.scryfall.com/bulk-data/default-cards') as response:
                metadata = json.load(response)
            source = Path(tmp) / 'default-cards.json'
            with get(metadata['download_uri']) as response, source.open('wb') as handle:
                shutil.copyfileobj(response, handle)
        with source.open() as handle:
            build(json.load(handle), args.out, not args.no_images)


if __name__ == '__main__':
    main()
