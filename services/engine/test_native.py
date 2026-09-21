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
            'engine': session.verification_fingerprint(),
            'observation': current['observation'],
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
            'engine': probe.verification_fingerprint(),
        }
        actual['observation'] = observed['observation']
        if (actual['public']['sha256'] != expected[index]['public']['sha256'] or
                actual['engine'] != expected[index]['engine'] or
                actual['observation'] != expected[index]['observation']):
            diagnostic = {'offset': index, 'expected': expected[index], 'actual': actual}
            Path('engine-native-divergence.json').write_text(json.dumps(diagnostic, indent=2))
            print(json.dumps(diagnostic, indent=2), flush=True)
            raise ReplayDivergenceError('Native reconstruction diverged at decision %d; see diagnostic.' % index)
        probe.step_action_ids(step['actionIds'], step['fingerprint'])
finally:
    probe.close()


def deterministic_trace(spec, decisions=80, selector=None):
    session = NativeSession(spec)
    agent = make_agent('random', seed=99173)
    trace = []
    try:
        for _ in range(decisions):
            if session.finished:
                break
            current = session.observation()
            observation = current['observation']
            selection = selector(observation) if selector is not None else agent.select(observation)
            options = observation['select'].get('option', [])
            selected_action_ids = [options[index]['actionId'] for index in selection]
            trace.append({
                'public': current['fingerprint'],
                'engine': session.verification_fingerprint(),
                'selectType': observation['select'].get('type'),
                'options': [
                    (option.get('actionId'), option.get('type'), option.get('label'))
                    for option in options
                ],
                'selection': list(selection),
                'actionIds': selected_action_ids,
                'observation': observation,
            })
            session.step(selection, current['fingerprint'])
        terminal = session.finished
        final_result = session.observation()['result'] if terminal else None
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
assert all('engineFingerprint' not in row['observation'] for row in trace_a), \
    'Private verification state leaked into an agent observation.'
assert terminal_a == terminal_b == terminal_c
assert result_a == result_b == result_c
uuid_pattern = __import__('re').compile(
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}')
assert not uuid_pattern.search(json.dumps(trace_a)), \
    'Canonical agent trace leaked a process-local UUID.'
assert not uuid_pattern.search(json.dumps(result_a)), \
    'Canonical terminal result leaked a process-local UUID.'
assert any(row['selectType'] == 'PRIORITY' and len(row['options']) > 1 for row in trace_a), \
    'Determinism trace never reached a priority state with a non-pass action.'

other_seed = dict(play_spec, seed=20260922)
other_trace, _, _ = deterministic_trace(other_seed, decisions=1)
assert other_trace and trace_a
assert other_trace[0]['engine'] != trace_a[0]['engine'], \
    'Different shuffle seed unexpectedly produced the same hidden-state digest.'


def exercise_priority_action(observation):
    select = observation.get('select') or {}
    options = select.get('option') or []
    minimum = select.get('minCount', 0)
    if not options:
        return []
    prompt = select.get('type')
    if prompt == 'MULLIGAN':
        return [0]  # keep; this fixture is about in-game action semantics
    if prompt == 'PRIORITY':
        for index, option in enumerate(options):
            if option.get('type') != 'PASS_PRIORITY':
                return [index]
        return [0]
    if prompt == 'PAY_MANA':
        for index, option in enumerate(options):
            if option.get('type') != 'PROMPT_CANCEL_PAYMENT':
                return [index]
    if minimum and minimum > 0:
        return list(range(min(minimum, len(options))))
    return []


spell_spec = {
    'decks': [
        '24 Mountain\n36 Lightning Bolt',
        '24 Mountain\n36 Lightning Bolt',
    ],
    'seed': 20260923,
    'maxTurns': 6,
}
spell_a, spell_terminal_a, spell_result_a = deterministic_trace(
    spell_spec, decisions=120, selector=exercise_priority_action)
spell_b, spell_terminal_b, spell_result_b = deterministic_trace(
    spell_spec, decisions=120, selector=exercise_priority_action)
assert spell_a == spell_b, \
    'Targeted-spell fixture did not reproduce the same semantic and hidden-state trace.'
assert spell_terminal_a == spell_terminal_b
assert spell_result_a == spell_result_b
assert any(
    any(option[0] == 'CAST_SPELL' for option in row['options'])
    for row in spell_a
), 'Targeted-spell fixture never exposed a cast-spell legal action.'
assert any(row['selectType'] == 'TARGET' for row in spell_a), \
    'Targeted-spell fixture never exercised an engine target prompt.'


def exercise_combat_action(observation):
    select = observation.get('select') or {}
    options = select.get('option') or []
    prompt = select.get('type')
    if prompt == 'MULLIGAN':
        return [0]
    if prompt == 'PRIORITY':
        for wanted in ('PLAY_LAND', 'CAST_SPELL'):
            for index, option in enumerate(options):
                if option.get('type') == wanted:
                    return [index]
        return [0] if options else []
    if prompt == 'PAY_MANA':
        for index, option in enumerate(options):
            if option.get('type') != 'PROMPT_CANCEL_PAYMENT':
                return [index]
        return []
    if prompt == 'DECLARE_ATTACKERS':
        return [0] if options else []
    if prompt == 'DECLARE_BLOCKERS':
        return []
    minimum = select.get('minCount', 0)
    return list(range(min(minimum, len(options)))) if minimum else []


combat_spec = {
    'decks': [
        '24 Forest\n36 Grizzly Bears',
        '24 Forest\n36 Grizzly Bears',
    ],
    'seed': 20260924,
    'maxTurns': 8,
}
combat_a, combat_terminal_a, combat_result_a = deterministic_trace(
    combat_spec, decisions=160, selector=exercise_combat_action)
combat_b, combat_terminal_b, combat_result_b = deterministic_trace(
    combat_spec, decisions=160, selector=exercise_combat_action)
assert combat_a == combat_b, \
    'Combat fixture did not reproduce the same canonical and hidden-state trace.'
assert combat_terminal_a == combat_terminal_b
assert combat_result_a == combat_result_b
assert any(
    row['selectType'] == 'DECLARE_ATTACKERS' and row['actionIds']
    for row in combat_a
), 'Combat fixture never selected an attacker.'


branch_source = NativeSession(play_spec)
branch_checkpoint = None
branch_action_ids = None
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
        if (select.get('type') == 'PRIORITY'
                and minimum == 1 and maximum == 1 and len(options) >= 2):
            branch_checkpoint = branch_source.checkpoint()
            branch_action_ids = [options[-1]['actionId']]
            break
        branch_source.step(
            branch_agent.select(current['observation']),
            current['fingerprint'])
finally:
    branch_source.close()

assert branch_checkpoint is not None and branch_action_ids is not None, \
    'Could not find a deterministic single-choice branch root.'

branch_a = NativeSession.restore(branch_checkpoint)
branch_b = NativeSession.restore(branch_checkpoint)
try:
    before_a = branch_a.observation()
    before_b = branch_b.observation()
    assert before_a['fingerprint'] == before_b['fingerprint']
    assert branch_a.verification_fingerprint() == branch_b.verification_fingerprint()
    assert before_a['observation'] == before_b['observation']
    after_a = branch_a.step_action_ids(branch_action_ids, before_a['fingerprint'])
    after_b = branch_b.step_action_ids(branch_action_ids, before_b['fingerprint'])
    assert after_a.get('finished') == after_b.get('finished')
    if not after_a.get('finished'):
        assert after_a['fingerprint'] == after_b['fingerprint']
        assert branch_a.verification_fingerprint() == branch_b.verification_fingerprint()
        assert after_a['observation'] == after_b['observation']
    else:
        assert after_a['result'] == after_b['result']
finally:
    branch_a.close()
    branch_b.close()

restored = NativeSession.restore(checkpoint)
try:
    rebuilt = restored.observation()
    assert rebuilt['fingerprint'] == checkpoint['root']['fingerprint']
    assert restored.verification_fingerprint() == checkpoint['root']['engineFingerprint']
    result = restored.autoplay(['random', 'first'], 200)
    Path('engine-native-replay.jsonl').write_text('\n'.join(json.dumps(frame) for frame in result['frames']) + '\n')
    report = {'reference': checkpoint['revision'], 'restoredPrefix': len(checkpoint['steps']),
              'decisionsAfterRestore': len(result['frames']), 'terminal': result['state']['finished'],
              'elapsedSeconds': round(time.perf_counter() - started, 3),
              'verifiedHiddenRoot': True,
              'repeatTraceDecisions': len(trace_a),
              'repeatTraceRuns': 3,
              'targetedSpellTraceDecisions': len(spell_a),
              'targetedSpellRepeatRuns': 2,
              'verifiedTargetPrompt': any(row['selectType'] == 'TARGET' for row in spell_a),
              'combatTraceDecisions': len(combat_a),
              'combatRepeatRuns': 2,
              'verifiedAttackPrompt': any(
                  row['selectType'] == 'DECLARE_ATTACKERS' and row['actionIds']
                  for row in combat_a),
              'verifiedAlternativePriorityBranch': True,
              'verifiedSemanticActionIds': all(
                  'actionIds' in step for step in checkpoint['steps']),
              'verifiedCanonicalAgentObservations': True,
              'verifiedNoRuntimeUuidLeak': True,
              'interpretation': 'A deterministic replay/restore smoke test over semantic public actions and a private hidden-state digest on creature/combat and targeted-spell paths; not complete card/rules equivalence or a throughput benchmark.'}
    Path('engine-native-result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    restored.close()
