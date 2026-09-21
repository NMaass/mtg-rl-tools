import json
import time
from pathlib import Path
from native import NativeSession
from magic_cabt.agents import make_agent
from magic_cabt.search.replay_search import ReplayDivergenceError, observation_signature

spec = {'decks': ['24 Forest\n36 Grizzly Bears', '24 Forest\n36 Grizzly Bears'], 'seed': 7, 'maxTurns': 5}
started = time.perf_counter()
session = NativeSession(spec)
expected = []
try:
    agent = make_agent('first', seed=7)
    for _ in range(12):
        if session.finished:
            break
        current = session.observation()
        expected.append({
            'public': observation_signature(current['observation']),
            'engine': current['engineFingerprint'],
        })
        session.step(agent.select(current['observation']), current['fingerprint'])
    checkpoint = session.checkpoint()
    assert not session.finished, 'Fixture ended before the restore boundary.'
    before = session.observation()
    try:
        session.step([0], 'stale')
        raise AssertionError('Stale step was accepted')
    except ReplayDivergenceError:
        pass
    assert session.observation() == before
finally:
    session.close()

probe = NativeSession(spec)
try:
    for index, step in enumerate(checkpoint['steps']):
        observed = probe.observation()
        actual = {
            'public': observation_signature(observed['observation']),
            'engine': observed['engineFingerprint'],
        }
        if (actual['public']['sha256'] != expected[index]['public']['sha256'] or
                actual['engine'] != expected[index]['engine']):
            diagnostic = {'offset': index, 'expected': expected[index], 'actual': actual}
            Path('engine-native-divergence.json').write_text(json.dumps(diagnostic, indent=2))
            print(json.dumps(diagnostic, indent=2), flush=True)
            raise ReplayDivergenceError('Native reconstruction diverged at decision %d; see diagnostic.' % index)
        probe.step(step['selection'], step['fingerprint'])
finally:
    probe.close()


def deterministic_trace(spec, decisions=80):
    session = NativeSession(spec)
    agent = make_agent('random', seed=99173)
    trace = []
    try:
        for _ in range(decisions):
            if session.finished:
                break
            current = session.observation()
            observation = current['observation']
            selection = agent.select(observation)
            trace.append({
                'public': current['fingerprint'],
                'engine': current['engineFingerprint'],
                'selectType': observation['select'].get('type'),
                'options': [
                    (option.get('type'), option.get('label'))
                    for option in observation['select'].get('option', [])
                ],
                'selection': list(selection),
            })
            session.step(selection, current['fingerprint'])
        terminal = session.finished
        final_result = session.bridge.result if terminal else None
        return trace, terminal, final_result
    finally:
        session.close()


play_spec = {
    'decks': [
        '24 Forest\n36 Grizzly Bears',
        '24 Forest\n36 Grizzly Bears',
    ],
    'seed': 20260921,
    'maxTurns': 8,
}
trace_a, terminal_a, result_a = deterministic_trace(play_spec)
trace_b, terminal_b, result_b = deterministic_trace(play_spec)
trace_c, terminal_c, result_c = deterministic_trace(play_spec)
assert trace_a == trace_b == trace_c, 'Same seed/action policy did not reproduce the same semantic and hidden-state trace.'
assert terminal_a == terminal_b == terminal_c
assert result_a == result_b == result_c
assert any(row['selectType'] == 'PRIORITY' and len(row['options']) > 1 for row in trace_a), \
    'Determinism trace never reached a priority state with a non-pass action.'

other_seed = dict(play_spec, seed=20260922)
other_trace, _, _ = deterministic_trace(other_seed, decisions=1)
assert other_trace and trace_a
assert other_trace[0]['engine'] != trace_a[0]['engine'], \
    'Different shuffle seed unexpectedly produced the same hidden-state digest.'


branch_source = NativeSession(play_spec)
branch_checkpoint = None
branch_selection = None
branch_agent = make_agent('random', seed=4455)
try:
    for _ in range(120):
        if branch_source.finished:
            break
        current = branch_source.observation()
        select = current['observation']['select']
        options = select.get('option', [])
        minimum = select.get('minCount', 0)
        maximum = select.get('maxCount', len(options))
        if minimum == 1 and maximum == 1 and len(options) >= 2:
            branch_checkpoint = branch_source.checkpoint()
            branch_selection = [len(options) - 1]
            break
        branch_source.step(
            branch_agent.select(current['observation']),
            current['fingerprint'])
finally:
    branch_source.close()

assert branch_checkpoint is not None and branch_selection is not None, \
    'Could not find a deterministic single-choice branch root.'

branch_a = NativeSession.restore(branch_checkpoint)
branch_b = NativeSession.restore(branch_checkpoint)
try:
    before_a = branch_a.observation()
    before_b = branch_b.observation()
    assert before_a['fingerprint'] == before_b['fingerprint']
    assert before_a['engineFingerprint'] == before_b['engineFingerprint']
    after_a = branch_a.step(branch_selection, before_a['fingerprint'])
    after_b = branch_b.step(branch_selection, before_b['fingerprint'])
    assert after_a.get('finished') == after_b.get('finished')
    if not after_a.get('finished'):
        assert after_a['fingerprint'] == after_b['fingerprint']
        assert after_a['engineFingerprint'] == after_b['engineFingerprint']
    else:
        assert after_a['result'] == after_b['result']
finally:
    branch_a.close()
    branch_b.close()

restored = NativeSession.restore(checkpoint)
try:
    rebuilt = restored.observation()
    assert rebuilt['fingerprint'] == checkpoint['root']['fingerprint']
    assert rebuilt['engineFingerprint'] == checkpoint['root']['engineFingerprint']
    result = restored.autoplay(['random', 'first'], 200)
    Path('engine-native-replay.jsonl').write_text('\n'.join(json.dumps(frame) for frame in result['frames']) + '\n')
    report = {'reference': checkpoint['revision'], 'restoredPrefix': len(checkpoint['steps']),
              'decisionsAfterRestore': len(result['frames']), 'terminal': result['state']['finished'],
              'elapsedSeconds': round(time.perf_counter() - started, 3),
              'verifiedHiddenRoot': True,
              'repeatTraceDecisions': len(trace_a),
              'repeatTraceRuns': 3,
              'verifiedAlternativeBranch': True,
              'interpretation': 'A deterministic replay/restore smoke test over public actions and a private full-state digest; not complete card/rules equivalence or a throughput benchmark.'}
    Path('engine-native-result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    restored.close()
