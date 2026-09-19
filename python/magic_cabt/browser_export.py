"""Browser transport adapter over the existing parsers; not a second rules engine."""

import copy
import json
import re
from collections import defaultdict


def _sid(value):
    return str(value) if value is not None else None


def _number(value):
    return value if type(value) in (int, float) else None


def project_snapshot(snapshot, catalog=None, viewer=None):
    catalog = catalog or {}
    arena = catalog.get('arena', {})
    names = catalog.get('names', {})
    hero = _sid(snapshot.get('localSeat')) if viewer is None else str(viewer)
    hero = hero or 'public'
    zones = snapshot.get('zones') or {}
    raw_cards = []
    for zone in ('battlefield', 'stack', 'exile', 'command'):
        raw_cards.extend((obj, zone, None) for obj in zones.get(zone) or [])
    for owner, objects in (zones.get('graveyards') or {}).items():
        raw_cards.extend((obj, 'graveyard', str(owner)) for obj in objects)
    raw_cards.extend((obj, 'hand', hero) for obj in (zones.get('hands') or {}).get(hero, []))
    aliases, cards, counts = {}, [], defaultdict(int)
    for obj, zone, zone_owner in raw_cards:
        if not isinstance(obj, dict):
            continue
        hidden = bool(obj.get('faceDown') or obj.get('faceDownFlag'))
        meta = {} if hidden else arena.get(str(obj.get('grpId'))) or names.get(obj.get('name')) or {}
        name = 'Face-down card' if hidden else obj.get('name') or meta.get('name') or 'Unresolved card'
        counts[name] += 1
        identity = obj.get('instanceId')
        ref = '%s · %d' % (name, counts[name])
        if identity is not None:
            aliases[str(identity)] = ref
        owner = _sid(obj.get('ownerSeat')) or zone_owner or 'unknown'
        controller = _sid(obj.get('controllerSeat')) or owner
        raw_types = meta.get('types') or obj.get('cardTypes') or []
        card = {'ref': ref, 'name': name, 'zone': zone, 'owner': owner,
                'controller': controller, 'type': '' if hidden else ' '.join(map(str, raw_types)),
                'mana': '' if hidden else meta.get('manaCost', ''),
                'rules': '' if hidden else meta.get('oracleText', ''),
                'power': obj.get('power'), 'toughness': obj.get('toughness'),
                'tapped': bool(obj.get('tapped')), 'faceDown': hidden,
                'damage': _number(obj.get('damage')) or 0,
                'counters': {str(k): v for k, v in (obj.get('counters') or {}).items() if _number(v) is not None},
                'attacking': bool(obj.get('attacking') or obj.get('attackState') == 'AttackState_Attacking')}
        if not hidden and meta.get('art'):
            card['art'] = meta['art']
        cards.append((card, obj))
    for card, obj in cards:
        if obj.get('attachedTo') is not None:
            card['attachedTo'] = aliases.get(str(obj['attachedTo']), 'Unresolved object')
    players = []
    for player in snapshot.get('players') or []:
        seat = _sid(player.get('seat'))
        if not seat:
            continue
        players.append({'id': seat, 'life': _number(player.get('life')),
                        'handCount': _number(player.get('handCount')),
                        'libraryCount': _number(player.get('libraryCount')),
                        'hand': [c for c, _ in cards if c['zone'] == 'hand'] if seat == hero else [],
                        'revealedHand': []})
    if len(players) < 2:
        return None, aliases
    return {'viewer': hero, 'active': _sid(snapshot.get('activeSeat')),
            'priority': _sid(snapshot.get('prioritySeat')),
            'players': players, 'cards': [c for c, _ in cards if c['zone'] != 'hand'], 'known': []}, aliases


def project_decision(record, catalog=None):
    snapshot = (record.get('observation') or {}).get('current') or {}
    view, aliases = project_snapshot(snapshot, catalog)
    if view is None:
        return None
    prompt = (record.get('observation') or {}).get('select') or {}
    arena = (catalog or {}).get('arena', {})
    options = []
    for opt in prompt.get('option') or []:
        payload = opt.get('payload') or {}
        label = str(opt.get('label') or 'Unknown action')
        name = payload.get('sourceName') or (arena.get(str(payload.get('grpId'))) or {}).get('name')
        name = aliases.get(str(payload.get('instanceId')), name)
        kind = str(opt.get('type') or '').upper()
        if kind in ('PASS', 'PASS_PRIORITY'):
            label = 'Pass priority'
        elif name and ('CAST' in kind or 'SPELL' in kind):
            label = 'Cast ' + name
        elif name and 'LAND' in kind:
            label = 'Play ' + name
        elif name and ('ACTIVAT' in kind or 'MANA' in kind):
            label = 'Activate ' + name
        else:
            label = re.sub(r'grpId=\d+(?: instance=\d+)?', name or 'Unresolved card', label)
            label = re.sub(r'(?:instance|instanceId)=([\d]+)', lambda m: aliases.get(m[1], 'Unresolved object'), label)
        index = opt.get('index')
        if type(index) is not int:
            continue
        detail = str(payload.get('rule') or '')
        options.append({'id': 'option_%d' % index, 'label': label, 'detail': detail})
    if not options:
        return {'turn': int(snapshot.get('turnNumber') or 0), 'phase': str(snapshot.get('phase') or ''),
                'label': 'Unrecorded action space', 'views': {view['viewer']: view}, 'decisions': {},
                'warnings': ['No legal choices were captured for this decision.']}
    selected = record.get('select')
    matched = record.get('selectionMatched') is not False
    ids = {o['id'] for o in options}
    choice = ['option_%d' % i for i in selected] if isinstance(selected, list) and all(type(i) is int for i in selected) else None
    if choice is not None and any(i not in ids for i in choice):
        matched, choice = False, None
    kind = str(prompt.get('type') or '')
    supported = (matched and kind in ('ACTIONSAVAILABLEREQ', 'PRIORITY') and
                 prompt.get('minCount') == 1 and prompt.get('maxCount') == 1 and
                 view['priority'] == view['viewer'])
    decision = {'kind': kind, 'options': options, 'chosen': choice,
                'min': int(prompt.get('minCount') or 0), 'max': int(prompt.get('maxCount') or 0),
                'source': 'arena-captured', 'supported': supported}
    if not supported:
        decision['reason'] = 'This captured prompt is not a verified single-choice hero priority.'
    played = '; '.join(o['label'] for o in options if choice and o['id'] in choice)
    return {'turn': int(snapshot.get('turnNumber') or 0),
            'phase': ' / '.join(str(snapshot[k]) for k in ('phase', 'step') if snapshot.get(k)),
            'label': played or 'Decision recorded', 'views': {view['viewer']: view},
            'decisions': {view['viewer']: decision}, 'warnings': []}


def _key(snapshot):
    return snapshot.get('matchId'), snapshot.get('gameInstance'), snapshot.get('seq')


def arena_replays(text, catalog):
    from .arena_log import iter_log_entries
    from .arena_mirror.tracker import ArenaMatchTracker, StreamingNormalizer

    stream, positions = [], {}
    errors = 0

    def snapshot(current, event):
        item = {'snapshot': copy.deepcopy(current), 'decisions': []}
        positions[_key(current)] = len(stream)
        stream.append(item)

    def decision(record):
        current = (record.get('observation') or {}).get('current') or {}
        position = positions.get(_key(current))
        if position is None:
            stream.append({'snapshot': current, 'decisions': [copy.deepcopy(record)]})
        else:
            stream[position]['decisions'].append(copy.deepcopy(record))

    tracker = ArenaMatchTracker(on_snapshot=snapshot, on_decision=decision)
    normalizer = StreamingNormalizer()
    for entry in iter_log_entries(text.splitlines(True)):
        _, events, parse_errors = normalizer.feed(entry)
        errors += len(parse_errors)
        for event in events:
            tracker.handle_event(event)
    groups = {}
    for item in stream:
        current = item['snapshot']
        group = (current.get('matchId'), current.get('gameInstance'))
        frames = groups.setdefault(group, [])
        if item['decisions']:
            for record in item['decisions']:
                frame = project_decision(record, catalog)
                if frame:
                    frames.append(frame)
        else:
            view, _ = project_snapshot(current, catalog)
            if view:
                frames.append({'turn': int(current.get('turnNumber') or 0),
                               'phase': ' / '.join(str(current[k]) for k in ('phase', 'step') if current.get(k)),
                               'label': 'Position updated', 'views': {view['viewer']: view},
                               'decisions': {}, 'warnings': []})
    result = []
    for i, frames in enumerate(groups.values(), 1):
        if frames:
            warnings = ['Known-card memory is limited to the captured snapshot; no missing private information is inferred.']
            if errors:
                warnings.append('%d log chunks could not be parsed.' % errors)
            result.append({'version': 1, 'title': 'Arena game %d' % i, 'source': 'arena', 'frames': frames, 'warnings': warnings})
    if not result:
        raise ValueError('No replay states found. Enable Detailed Logs in Arena and record a game.')
    return result


def mtgo_replays(text, catalog):
    from .mtgo_video.parse import parse_log
    from .mtgo_video.state import GameSimulator

    if '\x00' in text:
        raise ValueError('Native binary MTGO files are not a text game log. Export or copy the game log as text.')
    events = parse_log([line for line in text.splitlines() if line.strip()])
    names = []
    for event in events:
        name = event.get('player')
        if name and name not in names:
            names.append(name)
    if len(names) != 2:
        raise ValueError('Select one text game log with two identifiable players.')
    def card_info(name):
        entry = (catalog.get('names') or {}).get(name) or {}
        return dict(entry, type_line=' '.join(entry.get('types') or []))
    simulator = GameSimulator(card_info=card_info)
    simulator.seed_players(names)
    frames = []
    unparsed = sum(event['type'] == 'UNPARSED' for event in events)
    for event in events:
        for snapshot in simulator.apply(event):
            snapshot['prioritySeat'] = None
            view, _ = project_snapshot(snapshot, catalog, viewer='public')
            if not view:
                continue
            label = str(event.get('text') or event['type'])
            for index, name in enumerate(names, 1):
                label = label.replace(name, 'Player %d' % index)
            frames.append({'turn': int(snapshot.get('turnNumber') or 0), 'phase': str(snapshot.get('phase') or ''),
                           'label': label, 'views': {'public': view}, 'decisions': {}, 'warnings': []})
    if not frames:
        raise ValueError('No supported MTGO text-game events found.')
    warnings = ['Reconstructed public MTGO log. Not an exact rules replay. No private hands or legal priority choices are available.',
                '%d unparsed entries; %d reconstruction warnings.' % (unparsed, len(simulator.warnings))]
    return [{'version': 1, 'title': 'MTGO public game', 'source': 'mtgo', 'frames': frames, 'warnings': warnings}]


def parse_export(text, kind, catalog_json='{}'):
    catalog = json.loads(catalog_json)
    if not isinstance(catalog, dict):
        raise ValueError('Invalid card catalogue.')
    if kind == 'arena':
        result = arena_replays(text, catalog)
    elif kind == 'mtgo':
        result = mtgo_replays(text, catalog)
    else:
        raise ValueError('Unknown log format.')
    return json.dumps(result, ensure_ascii=True, allow_nan=False)
