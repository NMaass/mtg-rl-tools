import copy
import json
import time
from pathlib import Path
from native import NativeSession
from magic_cabt.agents import make_agent
from magic_cabt.search.replay_search import ReplayDivergenceError

spec={'decks':['24 Forest\n36 Grizzly Bears','24 Forest\n36 Grizzly Bears'],'seed':7,'maxTurns':5}
start=time.perf_counter()
session=NativeSession(spec)
try:
    agent=make_agent('first',seed=7)
    for _ in range(12):
        if session.finished:break
        current=session.observation()
        session.step(agent.select(current['observation']),current['fingerprint'])
    checkpoint=session.checkpoint()
    assert not session.finished, 'Fixture ended before the restore boundary.'
    before=session.observation()
    try:
        session.step([0],'stale')
        raise AssertionError('Stale step was accepted')
    except ReplayDivergenceError:
        pass
    assert session.observation()==before
finally:
    session.close()
restored=NativeSession.restore(checkpoint)
try:
    assert restored.observation()['fingerprint']==checkpoint['root']['fingerprint']
    result=restored.autoplay(['random','first'],200)
    Path('engine-native-replay.jsonl').write_text('\n'.join(json.dumps(f) for f in result['frames'])+'\n')
    report={'reference':checkpoint['revision'],'restoredPrefix':len(checkpoint['steps']),
            'decisionsAfterRestore':len(result['frames']),'terminal':result['state']['finished'],
            'elapsedSeconds':round(time.perf_counter()-start,3),
            'interpretation':'A bounded smoke test, not complete card/rules equivalence or a throughput benchmark.'}
    Path('engine-native-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
finally:
    restored.close()
