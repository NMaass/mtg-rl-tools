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
        expected.append(observation_signature(current['observation']))
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
        actual = observation_signature(probe.observation()['observation'])
        if actual['sha256'] != expected[index]['sha256']:
            diagnostic = {'offset': index, 'expected': expected[index], 'actual': actual}
            Path('engine-native-divergence.json').write_text(json.dumps(diagnostic, indent=2))
            print(json.dumps(diagnostic, indent=2), flush=True)
            raise ReplayDivergenceError('Native reconstruction diverged at decision %d; see diagnostic.' % index)
        probe.step(step['selection'], step['fingerprint'])
finally:
    probe.close()

restored = NativeSession.restore(checkpoint)
try:
    assert restored.observation()['fingerprint'] == checkpoint['root']['fingerprint']
    result = restored.autoplay(['random', 'first'], 200)
    Path('engine-native-replay.jsonl').write_text('\n'.join(json.dumps(frame) for frame in result['frames']) + '\n')
    report = {'reference': checkpoint['revision'], 'restoredPrefix': len(checkpoint['steps']),
              'decisionsAfterRestore': len(result['frames']), 'terminal': result['state']['finished'],
              'elapsedSeconds': round(time.perf_counter() - started, 3),
              'interpretation': 'A bounded smoke test, not complete card/rules equivalence or a throughput benchmark.'}
    Path('engine-native-result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    restored.close()
