import json
import time
from pathlib import Path

from native import NativeSession
from magic_cabt.agents import make_agent
from magic_cabt.search.replay_search import ReplayDivergenceError

BASE_DECK = '24 Forest\n36 Grizzly Bears'
SEEDS = (1, 7, 73)
TRACE_DECISIONS = 24
CONTINUATION_DECISIONS = 20
started = time.perf_counter()


def fail(kind, **data):
    diagnostic = {'kind': kind, **data}
    Path('engine-native-divergence.json').write_text(json.dumps(diagnostic, indent=2))
    print(json.dumps(diagnostic, indent=2), flush=True)
    raise ReplayDivergenceError(kind + '; see engine-native-divergence.json')


def spec(seed):
    return {'decks': [BASE_DECK, BASE_DECK], 'seed': seed, 'maxTurns': 8}


def capture(seed):
    session = NativeSession(spec(seed))
    agent = make_agent('first', seed=seed)
    trace = []
    branch_root = None
    try:
        for _ in range(TRACE_DECISIONS):
            if session.finished:
                break
            current = session.observation()
            hidden = session.verification()
            selection = agent.select(current['observation'])
            trace.append({
                'public': current['fingerprint'],
                'hidden': hidden['sha256'],
                'selection': list(selection),
                'observation': current['observation'],
            })
            select = current['observation']['select']
            if (branch_root is None and select.get('type') == 'PRIORITY'
                    and len(select.get('option') or []) >= 2):
                branch_root = session.checkpoint()
            session.step(selection, current['fingerprint'])
        checkpoint = session.checkpoint() if not session.finished else None
        return trace, branch_root, checkpoint
    finally:
        session.close()


def verify_trace(seed, trace):
    session = NativeSession(spec(seed))
    try:
        for offset, expected in enumerate(trace):
            current = session.observation()
            hidden = session.verification()
            if current['fingerprint'] != expected['public'] or hidden['sha256'] != expected['hidden']:
                fail('fresh reconstruction diverged', seed=seed, offset=offset,
                     expectedPublic=expected['public'], actualPublic=current['fingerprint'],
                     expectedHidden=expected['hidden'], actualHidden=hidden['sha256'],
                     expectedHiddenPayload=expected.get('hiddenPayload'),
                     actualHiddenPayload=hidden.get('payload'))
            session.step(expected['selection'], current['fingerprint'])
    finally:
        session.close()


def assert_same_continuation(checkpoint, selection):
    left = NativeSession.branch(checkpoint, selection)
    right = NativeSession.branch(checkpoint, selection)
    agent = make_agent('first', seed=991)
    compared = 0
    try:
        for _ in range(CONTINUATION_DECISIONS):
            if left.finished or right.finished:
                if left.finished != right.finished or left.bridge.result != right.bridge.result:
                    fail('same-action branches disagree at terminal state',
                         left=left.bridge.result, right=right.bridge.result)
                break
            a, b = left.observation(), right.observation()
            ah, bh = left.verification(), right.verification()
            if a['fingerprint'] != b['fingerprint'] or ah['sha256'] != bh['sha256']:
                fail('same-action branches diverged', offset=compared,
                     leftPublic=a['fingerprint'], rightPublic=b['fingerprint'],
                     leftHidden=ah, rightHidden=bh)
            chosen = agent.select(a['observation'])
            left.step(chosen, a['fingerprint'])
            right.step(chosen, b['fingerprint'])
            compared += 1
        return compared, left.observation(), left.verification() if not left.finished else None
    finally:
        left.close()
        right.close()


all_traces = {}
branch_root = None
for seed in SEEDS:
    trace, candidate_root, checkpoint = capture(seed)
    assert trace, 'Determinism trace is empty.'
    verify_trace(seed, trace)
    if checkpoint is not None:
        restored = NativeSession.restore(checkpoint)
        try:
            assert restored.observation()['fingerprint'] == checkpoint['root']['fingerprint']
            assert restored.verification()['sha256'] == checkpoint['verification']['sha256']
        finally:
            restored.close()
    all_traces[seed] = trace
    if seed == 7:
        branch_root = candidate_root

assert branch_root is not None, 'No multi-action priority root was found.'
root_options = branch_root['root']['observation']['select']['option']
assert len(root_options) >= 2
same_selection = [1]
other_selection = [0]

same_a = NativeSession.branch(branch_root, same_selection)
other = NativeSession.branch(branch_root, other_selection)
try:
    same_public = same_a.observation()['fingerprint']
    same_hidden = same_a.verification()['sha256']
    other_public = other.observation()['fingerprint']
    other_hidden = other.verification()['sha256']
    if same_public == other_public and same_hidden == other_hidden:
        fail('different legal actions produced an indistinguishable successor',
             root=branch_root['root']['fingerprint'],
             actionA=root_options[1], actionB=root_options[0])
finally:
    same_a.close()
    other.close()

compared, _, _ = assert_same_continuation(branch_root, same_selection)

# Stale roots must never mutate a live session.
stale = NativeSession.restore(branch_root)
try:
    before_public = stale.observation()
    before_hidden = stale.verification()
    try:
        stale.step(other_selection, 'stale')
        raise AssertionError('Stale step was accepted')
    except ReplayDivergenceError:
        pass
    assert stale.observation() == before_public
    assert stale.verification()['sha256'] == before_hidden['sha256']
finally:
    stale.close()

frames = [
    {'gameId': 'determinism-seed-7', 'sequence': index,
     'observation': item['observation'], 'selected': item['selection'],
     'publicFingerprint': item['public'], 'hiddenFingerprint': item['hidden']}
    for index, item in enumerate(all_traces[7])
]
Path('engine-native-replay.jsonl').write_text(
    '\n'.join(json.dumps(frame) for frame in frames) + '\n')

report = {
    'reference': branch_root['revision'],
    'setup': branch_root['setup'],
    'checkpointVersion': branch_root['version'],
    'seedsVerified': list(SEEDS),
    'decisionsPerSeed': {str(seed): len(trace) for seed, trace in all_traces.items()},
    'branchRootPosition': branch_root['root']['position'],
    'branchAction': root_options[1].get('label'),
    'alternateAction': root_options[0].get('label'),
    'sameActionContinuationDecisions': compared,
    'publicAndHiddenStateMatched': True,
    'differentActionDiverged': True,
    'elapsedSeconds': round(time.perf_counter() - started, 3),
    'interpretation': (
        'Verified deterministic transcript reconstruction and branching for the '
        'declared corpus against the pinned XMage build. This does not prove '
        'every Magic card/mechanic or bitwise JVM-state equivalence.'
    ),
}
Path('engine-native-result.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
