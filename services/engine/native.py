"""Bounded native agent interface over the declared XMage reference build."""

import copy
import json
import sys
from magic_cabt import CabtBridge
from magic_cabt.agents import make_agent, is_legal_selection
from magic_cabt.protocol import CabtProtocolError
from magic_cabt.search.replay_search import observation_signature, ReplayDivergenceError

REFERENCE = 'fd40ad5c29a92cef824cf12ba6d0e4daa25db975'
SETUP = 'declared-library-order-v1'
MAX_STEPS = 2000


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
        if self.finished:
            raise ValueError('The game has ended.')
        if len(self.steps) >= MAX_STEPS:
            raise ValueError('Decision budget exhausted.')
        current = self.observation()
        if fingerprint != current['fingerprint']:
            raise ReplayDivergenceError('Stale observation: no action was applied.')
        if not is_legal_selection(selection, self.response['observation']['select']):
            raise ValueError('Selection is not legal for the current engine prompt.')
        try:
            self.response = self.bridge.game_select(selection)
        except Exception:
            self.failed = True
            self.close()
            raise
        self.steps.append({'fingerprint': fingerprint, 'selection': list(selection)})
        return self.observation()

    def checkpoint(self):
        if self.finished:
            raise ValueError('A terminal game is a result, not a resumable root.')
        return {'kind': 'native-xmage-replay-root', 'version': 2,
                'revision': REFERENCE, 'setup': SETUP,
                'spec': copy.deepcopy(self.spec), 'steps': copy.deepcopy(self.steps),
                'root': self.observation()}

    @classmethod
    def restore(cls, checkpoint, bridge_factory=CabtBridge):
        if (not isinstance(checkpoint, dict) or
                checkpoint.get('kind') != 'native-xmage-replay-root' or
                checkpoint.get('version') != 2 or
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
                session.step(step['selection'], step['fingerprint'])
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
                    result = session.step(request['selection'], request['fingerprint'])
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
