"""Bounded native agent interface over the declared XMage reference build."""

import copy
import json
import re
import sys
from magic_cabt import CabtBridge
from magic_cabt.agents import make_agent, is_legal_selection
from magic_cabt.protocol import CabtProtocolError
from magic_cabt.search.replay_search import observation_signature, ReplayDivergenceError

REFERENCE = 'fd40ad5c29a92cef824cf12ba6d0e4daa25db975'
SETUP = 'declared-library-order-v1'
MAX_STEPS = 2000
_UUID = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}
    def __init__(self, spec, bridge_factory=CabtBridge):
        if not isinstance(spec, dict) or not isinstance(spec.get('decks'), list) or len(spec['decks']) != 2:
            raise ValueError('Provide exactly two deck lists.')
        if len(json.dumps(spec)) > 50000:
            raise ValueError('Deck specification is too large.')
        seed = spec.get('seed', 7)
        turns = spec.get('maxTurns', 20)
        if type(seed) is not int or type(turns) is not int or not 1 <= turns <= 200:
            raise ValueError('Provide an integer seed and maxTurns between 1 and 200.')
        self.spec = {'decks': copy.deepcopy(spec['decks']), 'seed': seed, 'maxTurns': turns}
        self.bridge = bridge_factory()
        self.steps = []
        self.failed = False
        try:
            self.response = self.bridge.game_start(*self.spec['decks'], seed=seed, max_turns=turns)
        except Exception:
            self.close()
            raise

    @property
    def finished(self):
        return self.bridge.finished

    def observation(self):
        if self.failed:
            raise ValueError('This session failed and cannot be continued.')
        if self.finished:
            return {'finished': True, 'result': canonical_result(self.bridge.result)}
        observation = canonical_observation(self.response['observation'])
        return {'finished': False, 'revision': REFERENCE, 'setup': SETUP,
                'position': len(self.steps),
                'fingerprint': observation_signature(observation)['sha256'],
                'engineFingerprint': self.bridge.engine_fingerprint(),
                'observation': copy.deepcopy(observation)}

    def step(self, selection, fingerprint):
        current = self.observation()
        select = current['observation']['select']
        if not is_legal_selection(selection, select):
            raise ValueError('Selection is not legal for the current engine prompt.')
        options = select.get('option') or []
        try:
            action_ids = [options[index]['actionId'] for index in selection]
        except (IndexError, KeyError, TypeError):
            raise ValueError('Current prompt has no stable action id for that selection.')
        return self.step_action_ids(action_ids, fingerprint)

    def step_action_ids(self, action_ids, fingerprint):
        if self.finished:
            raise ValueError('The game has ended.')
        if len(self.steps) >= MAX_STEPS:
            raise ValueError('Decision budget exhausted.')
        current = self.observation()
        if fingerprint != current['fingerprint']:
            raise ReplayDivergenceError('Stale observation: no action was applied.')
        select = current['observation']['select']
        options = select.get('option') or []
        index_by_id = {}
        for index, option in enumerate(options):
            action_id = option.get('actionId')
            if not isinstance(action_id, str) or not action_id:
                raise ValueError('Current prompt contains an option without a stable action id.')
            if action_id in index_by_id:
                raise ReplayDivergenceError('Current prompt contains duplicate stable action ids.')
            index_by_id[action_id] = index
        if not isinstance(action_ids, list) or any(
                not isinstance(action_id, str) or action_id not in index_by_id
                for action_id in action_ids):
            raise ReplayDivergenceError('Recorded action id is absent from the current prompt.')
        selection = [index_by_id[action_id] for action_id in action_ids]
        if not is_legal_selection(selection, select):
            raise ValueError('Action-id selection violates the current prompt bounds.')
        try:
            self.response = self.bridge.game_select_ids(action_ids)
        except Exception:
            self.failed = True
            self.close()
            raise
        self.steps.append({
            'fingerprint': fingerprint,
            'actionIds': list(action_ids),
            'selection': selection,
        })
        return self.observation()

    def checkpoint(self):
        if self.finished:
            raise ValueError('A terminal game is a result, not a resumable root.')
        return {'kind': 'native-xmage-replay-root', 'version': 3,
                'revision': REFERENCE, 'setup': SETUP,
                'spec': copy.deepcopy(self.spec), 'steps': copy.deepcopy(self.steps),
                'root': self.observation()}

    @classmethod
    def restore(cls, checkpoint, bridge_factory=CabtBridge):
        if (not isinstance(checkpoint, dict) or
                checkpoint.get('kind') != 'native-xmage-replay-root' or
                checkpoint.get('version') != 3 or
                checkpoint.get('revision') != REFERENCE or
                checkpoint.get('setup') != SETUP):
            raise ValueError('Only a native root from this engine and ordered-setup protocol can be restored.')
        steps = checkpoint.get('steps')
        expected = checkpoint.get('root')
        if (not isinstance(steps, list) or len(steps) > MAX_STEPS or
                not isinstance(expected, dict) or expected.get('finished') is not False):
            raise ValueError('Invalid replay prefix or non-resumable root.')
        session = cls(checkpoint['spec'], bridge_factory)
        try:
            for step in steps:
                action_ids = step.get('actionIds')
                if not isinstance(action_ids, list):
                    raise ValueError('Replay step has no stable action ids.')
                session.step_action_ids(action_ids, step['fingerprint'])
            actual = session.observation()
            if (actual.get('finished') or
                    actual.get('fingerprint') != expected.get('fingerprint') or
                    actual.get('engineFingerprint') != expected.get('engineFingerprint')):
                raise ReplayDivergenceError(
                    'Rebuilt root does not match the recorded public semantics and private verification digest.',
                    expected={'public': expected.get('fingerprint'),
                              'engine': expected.get('engineFingerprint')},
                    actual={'public': actual.get('fingerprint'),
                            'engine': actual.get('engineFingerprint')})
            return session
        except Exception:
            session.close()
            raise

    def autoplay(self, agent_specs=('random', 'first'), decisions=100):
        if len(agent_specs) != 2 or any(agent not in ('random', 'first') for agent in agent_specs):
            raise ValueError('Bundled controls are random and first. External agents use step.')
        if type(decisions) is not int or not 1 <= decisions <= MAX_STEPS:
            raise ValueError('Invalid decision budget.')
        agents = [make_agent(name, seed=self.spec['seed'] + index) for index, name in enumerate(agent_specs)]
        frames = []
        for _ in range(decisions):
            if self.finished:
                break
            current = self.observation()
            observation = current['observation']
            seat = observation['select']['playerIndex']
            selection = agents[seat].select(observation)
            frames.append({'gameId': 'native', 'sequence': len(self.steps), 'player': seat,
                           'observation': observation, 'selected': selection})
            self.step(selection, current['fingerprint'])
        return {'frames': frames, 'state': self.observation()}

    def close(self):
        if hasattr(self, 'bridge'):
            self.bridge.close()


def main():
    session = None
    try:
        for line in sys.stdin:
            if len(line) > 2 * 1024 * 1024:
                print(json.dumps({'ok': False, 'error': 'Request too large.'}), flush=True)
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError('Request must be an object.')
                command = request.get('command')
                if command in ('new', 'restore'):
                    if session is not None:
                        session.close()
                    session = None
                    session = NativeSession(request['spec']) if command == 'new' else NativeSession.restore(request['checkpoint'])
                    result = session.observation()
                elif session is None:
                    raise ValueError('Start or restore a game first.')
                elif command == 'observe':
                    result = session.observation()
                elif command == 'step':
                    if 'actionIds' in request:
                        result = session.step_action_ids(
                            request['actionIds'], request['fingerprint'])
                    else:
                        result = session.step(
                            request['selection'], request['fingerprint'])
                elif command == 'checkpoint':
                    result = session.checkpoint()
                elif command == 'autoplay':
                    result = session.autoplay(request.get('agents', ['random', 'first']), request.get('decisions', 100))
                else:
                    raise ValueError('Unknown command.')
                print(json.dumps({'ok': True, 'data': result}), flush=True)
            except (ValueError, KeyError, TypeError, ReplayDivergenceError, CabtProtocolError, OSError) as error:
                print(json.dumps({'ok': False, 'error': str(error)}), flush=True)
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    main()
)
_RUNTIME_ID_FIELDS = {
    'playerId', 'activePlayerId', 'priorityPlayerId', 'objectId', 'sourceId',
    'ownerId', 'controllerId', 'targetId', 'abilityId', 'originalId',
    'attackerId', 'defenderId', 'blockerId', 'defendingPlayerId', 'modeId',
}


def _identity_map(observation):
    mapping = {}
    current = observation.get('current') or {}
    for player in current.get('players') or []:
        raw = player.get('playerId')
        seat = player.get('playerIndex')
        if isinstance(raw, str) and isinstance(seat, int):
            mapping[raw] = 'P%d' % seat

    def visit(value):
        if isinstance(value, dict):
            ref = value.get('ref')
            if isinstance(ref, dict):
                raw = ref.get('objectId')
                semantic = ref.get('semanticId')
                if isinstance(raw, str) and isinstance(semantic, str) and semantic:
                    mapping[raw] = semantic
            payload = value.get('payload')
            if isinstance(payload, dict):
                for raw_key, ref_key in (
                    ('sourceId', 'sourceRef'), ('targetId', 'targetRef'),
                    ('objectId', 'objectRef'), ('attackerId', 'attackerRef'),
                    ('defenderId', 'defenderRef'), ('blockerId', 'blockerRef'),
                    ('defendingPlayerId', 'defendingPlayerRef'),
                ):
                    raw = payload.get(raw_key)
                    semantic = payload.get(ref_key)
                    if isinstance(raw, str) and isinstance(semantic, str) and semantic:
                        mapping[raw] = semantic
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(observation)
    return mapping


def _canonical_value(value, identities, field=None):
    if isinstance(value, dict):
        result = {}
        for key, nested in value.items():
            if key == 'semanticId':
                result[key] = nested
                continue
            if key in _RUNTIME_ID_FIELDS:
                if nested is None:
                    result[key] = None
                elif isinstance(nested, str):
                    result[key] = identities.get(nested)
                else:
                    result[key] = None
                continue
            result[key] = _canonical_value(nested, identities, key)
        return result
    if isinstance(value, list):
        return [_canonical_value(item, identities, field) for item in value]
    if isinstance(value, str) and _UUID.match(value):
        # Runtime UUIDs are never part of the deterministic agent contract.
        # Resolve known object/player references and redact implementation-only
        # ability/mode ids rather than letting process-local entropy leak out.
        return identities.get(value)
    return value


def canonical_observation(observation):
    identities = _identity_map(observation)
    return _canonical_value(copy.deepcopy(observation), identities)


def canonical_result(result):
    if not isinstance(result, dict):
        return copy.deepcopy(result)
    final_state = result.get('finalState')
    if not isinstance(final_state, dict):
        return {'winner': result.get('winner'), 'finalState': final_state}
    synthetic = {'current': final_state, 'select': {'option': []}}
    canonical = canonical_observation(synthetic)['current']
    return {'winner': result.get('winner'), 'finalState': canonical}


class NativeSession:
    def __init__(self, spec, bridge_factory=CabtBridge):
        if not isinstance(spec, dict) or not isinstance(spec.get('decks'), list) or len(spec['decks']) != 2:
            raise ValueError('Provide exactly two deck lists.')
        if len(json.dumps(spec)) > 50000:
            raise ValueError('Deck specification is too large.')
        seed = spec.get('seed', 7)
        turns = spec.get('maxTurns', 20)
        if type(seed) is not int or type(turns) is not int or not 1 <= turns <= 200:
            raise ValueError('Provide an integer seed and maxTurns between 1 and 200.')
        self.spec = {'decks': copy.deepcopy(spec['decks']), 'seed': seed, 'maxTurns': turns}
        self.bridge = bridge_factory()
        self.steps = []
        self.failed = False
        try:
            self.response = self.bridge.game_start(*self.spec['decks'], seed=seed, max_turns=turns)
        except Exception:
            self.close()
            raise

    @property
    def finished(self):
        return self.bridge.finished

    def observation(self):
        if self.failed:
            raise ValueError('This session failed and cannot be continued.')
        if self.finished:
            return {'finished': True, 'result': self.bridge.result}
        observation = self.response['observation']
        return {'finished': False, 'revision': REFERENCE, 'setup': SETUP,
                'position': len(self.steps),
                'fingerprint': observation_signature(observation)['sha256'],
                'engineFingerprint': self.bridge.engine_fingerprint(),
                'observation': copy.deepcopy(observation)}

    def step(self, selection, fingerprint):
        current = self.observation()
        select = current['observation']['select']
        if not is_legal_selection(selection, select):
            raise ValueError('Selection is not legal for the current engine prompt.')
        options = select.get('option') or []
        try:
            action_ids = [options[index]['actionId'] for index in selection]
        except (IndexError, KeyError, TypeError):
            raise ValueError('Current prompt has no stable action id for that selection.')
        return self.step_action_ids(action_ids, fingerprint)

    def step_action_ids(self, action_ids, fingerprint):
        if self.finished:
            raise ValueError('The game has ended.')
        if len(self.steps) >= MAX_STEPS:
            raise ValueError('Decision budget exhausted.')
        current = self.observation()
        if fingerprint != current['fingerprint']:
            raise ReplayDivergenceError('Stale observation: no action was applied.')
        select = current['observation']['select']
        options = select.get('option') or []
        index_by_id = {}
        for index, option in enumerate(options):
            action_id = option.get('actionId')
            if not isinstance(action_id, str) or not action_id:
                raise ValueError('Current prompt contains an option without a stable action id.')
            if action_id in index_by_id:
                raise ReplayDivergenceError('Current prompt contains duplicate stable action ids.')
            index_by_id[action_id] = index
        if not isinstance(action_ids, list) or any(
                not isinstance(action_id, str) or action_id not in index_by_id
                for action_id in action_ids):
            raise ReplayDivergenceError('Recorded action id is absent from the current prompt.')
        selection = [index_by_id[action_id] for action_id in action_ids]
        if not is_legal_selection(selection, select):
            raise ValueError('Action-id selection violates the current prompt bounds.')
        try:
            self.response = self.bridge.game_select_ids(action_ids)
        except Exception:
            self.failed = True
            self.close()
            raise
        self.steps.append({
            'fingerprint': fingerprint,
            'actionIds': list(action_ids),
            'selection': selection,
        })
        return self.observation()

    def checkpoint(self):
        if self.finished:
            raise ValueError('A terminal game is a result, not a resumable root.')
        return {'kind': 'native-xmage-replay-root', 'version': 3,
                'revision': REFERENCE, 'setup': SETUP,
                'spec': copy.deepcopy(self.spec), 'steps': copy.deepcopy(self.steps),
                'root': self.observation()}

    @classmethod
    def restore(cls, checkpoint, bridge_factory=CabtBridge):
        if (not isinstance(checkpoint, dict) or
                checkpoint.get('kind') != 'native-xmage-replay-root' or
                checkpoint.get('version') != 3 or
                checkpoint.get('revision') != REFERENCE or
                checkpoint.get('setup') != SETUP):
            raise ValueError('Only a native root from this engine and ordered-setup protocol can be restored.')
        steps = checkpoint.get('steps')
        expected = checkpoint.get('root')
        if (not isinstance(steps, list) or len(steps) > MAX_STEPS or
                not isinstance(expected, dict) or expected.get('finished') is not False):
            raise ValueError('Invalid replay prefix or non-resumable root.')
        session = cls(checkpoint['spec'], bridge_factory)
        try:
            for step in steps:
                action_ids = step.get('actionIds')
                if not isinstance(action_ids, list):
                    raise ValueError('Replay step has no stable action ids.')
                session.step_action_ids(action_ids, step['fingerprint'])
            actual = session.observation()
            if (actual.get('finished') or
                    actual.get('fingerprint') != expected.get('fingerprint') or
                    actual.get('engineFingerprint') != expected.get('engineFingerprint')):
                raise ReplayDivergenceError(
                    'Rebuilt root does not match the recorded public semantics and private verification digest.',
                    expected={'public': expected.get('fingerprint'),
                              'engine': expected.get('engineFingerprint')},
                    actual={'public': actual.get('fingerprint'),
                            'engine': actual.get('engineFingerprint')})
            return session
        except Exception:
            session.close()
            raise

    def autoplay(self, agent_specs=('random', 'first'), decisions=100):
        if len(agent_specs) != 2 or any(agent not in ('random', 'first') for agent in agent_specs):
            raise ValueError('Bundled controls are random and first. External agents use step.')
        if type(decisions) is not int or not 1 <= decisions <= MAX_STEPS:
            raise ValueError('Invalid decision budget.')
        agents = [make_agent(name, seed=self.spec['seed'] + index) for index, name in enumerate(agent_specs)]
        frames = []
        for _ in range(decisions):
            if self.finished:
                break
            current = self.observation()
            observation = current['observation']
            seat = observation['select']['playerIndex']
            selection = agents[seat].select(observation)
            frames.append({'gameId': 'native', 'sequence': len(self.steps), 'player': seat,
                           'observation': observation, 'selected': selection})
            self.step(selection, current['fingerprint'])
        return {'frames': frames, 'state': self.observation()}

    def close(self):
        if hasattr(self, 'bridge'):
            self.bridge.close()


def main():
    session = None
    try:
        for line in sys.stdin:
            if len(line) > 2 * 1024 * 1024:
                print(json.dumps({'ok': False, 'error': 'Request too large.'}), flush=True)
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError('Request must be an object.')
                command = request.get('command')
                if command in ('new', 'restore'):
                    if session is not None:
                        session.close()
                    session = None
                    session = NativeSession(request['spec']) if command == 'new' else NativeSession.restore(request['checkpoint'])
                    result = session.observation()
                elif session is None:
                    raise ValueError('Start or restore a game first.')
                elif command == 'observe':
                    result = session.observation()
                elif command == 'step':
                    if 'actionIds' in request:
                        result = session.step_action_ids(
                            request['actionIds'], request['fingerprint'])
                    else:
                        result = session.step(
                            request['selection'], request['fingerprint'])
                elif command == 'checkpoint':
                    result = session.checkpoint()
                elif command == 'autoplay':
                    result = session.autoplay(request.get('agents', ['random', 'first']), request.get('decisions', 100))
                else:
                    raise ValueError('Unknown command.')
                print(json.dumps({'ok': True, 'data': result}), flush=True)
            except (ValueError, KeyError, TypeError, ReplayDivergenceError, CabtProtocolError, OSError) as error:
                print(json.dumps({'ok': False, 'error': str(error)}), flush=True)
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    main()
