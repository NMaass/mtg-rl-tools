"""Safety, scoring, routing, tournament and promotion tests. No paid requests."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest

from magic_cabt.eval.play import acting_seat, play_game
from magic_cabt.heuristic import role
from magic_cabt.heuristic.experiments import classify_result, load_pool, schedule, summarize, play_one
from magic_cabt.heuristic.jev import Budget, JevAgent, payload_for
from magic_cabt.heuristic.policy import Config, HeuristicAgent, choose, selection_valid
from magic_cabt.heuristic.refinement import propose, regressions, promotion_gate
from magic_cabt.heuristic.review import render
from magic_cabt.heuristic.state import canonical, digest, project

ROOT = Path(__file__).resolve().parents[2]
SUITE = json.loads((ROOT / 'examples/heuristic/regressions.json').read_text())
POOL = load_pool(ROOT / 'examples/heuristic/deck_pool.json')


def example(name='land-before-pass'):
    return deepcopy(next(c['observation'] for c in SUITE['cases'] if c['id'] == name))


def fake_response(payload, key):
    keys = list(payload['questions']['action']['criteria'])
    return {'model':payload['model'], 'answers':{'action':{'type':'choice','choice':keys[-1],
            'probabilities':{k:(0.9 if k == keys[-1] else 0.1/(len(keys)-1)) for k in keys}}},
            'usage':{'cost':0.001,'input_tokens':50,'output_tokens':2}}


class ProjectionTest(unittest.TestCase):
    def test_hidden_and_future_information_excluded(self):
        ob=example()
        ob['current']['players'][1]['hand']=[{'name':'OPPONENT_HAND_SECRET'}]
        ob['current']['players'][1]['library']=[{'name':'FUTURE_DRAW_SECRET'}]
        ob['current']['winner']='RESULT_SECRET'
        ob['future']='FUTURE_STATE_SECRET'
        ob['select']['option'][1]['payload']['raw']='RAW_SECRET'
        text=canonical(project(ob,SUITE['catalog']).public_dict())
        self.assertNotIn('SECRET',text)
        self.assertIn('Forest',text)

    def test_face_down_identity_hidden(self):
        ob=example('attack-unblocked')
        ob['current']['battlefield'][0]['faceDown']=True
        ob['current']['battlefield'][0]['ref']['name']='Secret creature'
        ctx=project(ob,SUITE['catalog'])
        self.assertEqual(ctx.battlefield[0].name,'Face-down object')
        self.assertEqual(ctx.battlefield[0].text,'')

    def test_choosing_player_not_priority_player(self):
        ob=example();ob['current']['priorityPlayerId']='o'
        self.assertEqual(project(ob).hero,'h')

    def test_conflicting_select_identity_rejected(self):
        ob=example();ob['select']['playerId']='o'
        with self.assertRaises(ValueError):project(ob)

    def test_face_down_action_labels_cannot_leak_identity(self):
        ob=example('attack-unblocked')
        card=ob['current']['battlefield'][0]
        card['faceDown']=True;card['ref']['name']='SECRET_CREATURE'
        option=ob['select']['option'][0]
        option['label']='Attack with SECRET_CREATURE'
        option['payload']['attackerName']='SECRET_CREATURE'
        self.assertNotIn('SECRET_CREATURE',canonical(project(ob,SUITE['catalog']).public_dict()))

    def test_missing_identity_rejected(self):
        ob=example();ob['current']['players'][0].pop('playerId')
        with self.assertRaises(ValueError):project(ob)

    def test_arena_not_silently_misinterpreted(self):
        ob=example();ob['current']['zones']={}
        with self.assertRaises(ValueError):project(ob)

    def test_noncontiguous_indices_rejected(self):
        ob=example();ob['select']['option'][0]['index']=22
        with self.assertRaises(ValueError):project(ob)

    def test_bounds_above_option_count_are_capped(self):
        ob=example();ob['select']['maxCount']=5
        self.assertEqual(project(ob).maximum,2)

    def test_no_original_mutation(self):
        ob=example();before=deepcopy(ob)
        HeuristicAgent(catalog=SUITE['catalog']).select(ob)
        self.assertEqual(ob,before)


class ScoringTest(unittest.TestCase):
    def test_frozen_tactical_suite(self):
        result=regressions(Config(),SUITE)
        self.assertEqual(result['failed'],[])
        self.assertEqual(result['evaluated'],16)

    def test_all_five_components_present(self):
        _,report=HeuristicAgent().explain(example())
        self.assertEqual({c['module'] for c in report['options'][0]['components']},
                         {'role','threats','mana','tactics','plans'})

    def test_mandatory_negative_action_still_selected(self):
        ctx=project(example())
        self.assertEqual(choose(ctx,[-50,-20]),[1])

    def test_optional_negative_actions_declined(self):
        ctx=project(example('avoid-free-chump-attack'))
        self.assertEqual(choose(ctx,[-4]),[])

    def test_duplicate_blocker_combination_rejected(self):
        ob=example('no-duplicate-attacker');ob['select']['type']='DECLARE_BLOCKERS'
        for a in ob['select']['option']:a['payload']['blockerId']='b'
        ctx=project(ob)
        self.assertFalse(selection_valid(ctx,[0,1]))
        self.assertEqual(len(choose(ctx,[3,2])),1)

    def test_unknown_target_does_not_assume_removal(self):
        ob=example('do-not-harm-own-creature')
        for a in ob['select']['option']:a['payload'].pop('sourceRule')
        scores=HeuristicAgent(catalog=SUITE['catalog']).score(ob)
        self.assertEqual(scores,[0,0])

    def test_unsupported_prompt_reported(self):
        ob=example();ob['select']['type']='PILE'
        _,report=HeuristicAgent().explain(ob)
        self.assertTrue(report['fallback'])

    def test_roles_reassessed(self):
        ob=example('avoid-free-chump-attack');ob['current']['players'][0]['life']=4
        self.assertEqual(role.assign(project(ob)).name,'stabilize')

    def test_invalid_weights_rejected(self):
        for bad in (float('nan'),float('inf'),-1,5,True):
            weights=dict(Config().weights);weights['mana']=bad
            with self.assertRaises(ValueError):Config(weights)
        with self.assertRaises(ValueError):Config({'mana':1})

    def test_config_hash_changes_with_weight(self):
        weights=dict(Config().weights);weights['mana']=1.1
        self.assertNotEqual(Config().fingerprint,Config(weights).fingerprint)

    def test_selection_boolean_rejected(self):
        self.assertFalse(selection_valid(project(example()),[True]))


class JevTest(unittest.TestCase):
    def test_scores_exact_same_option_ids(self):
        ctx=project(example(),SUITE['catalog'])
        payload=payload_for(ctx)
        self.assertEqual(list(payload['questions']['action']['criteria']),['option_0','option_1'])

    def test_selective_high_margin_no_call(self):
        def forbidden(*args):self.fail('unexpected request')
        agent=JevAgent(HeuristicAgent(catalog=SUITE['catalog']),api_key='test-key',transport=forbidden)
        self.assertEqual(agent.select(example()),[1])
        self.assertEqual(agent.last_decision['jev']['reason'],'heuristic-margin')

    def test_disagreement_and_acceptance_recorded(self):
        ob=example('save-burn-without-target')
        agent=JevAgent(HeuristicAgent(catalog=SUITE['catalog']),api_key='test-key',mode='always',transport=fake_response)
        self.assertEqual(agent.select(ob),[1])
        self.assertTrue(agent.last_decision['jev']['disagreement'])
        self.assertEqual(agent.last_decision['jev']['provenance'],'fixture')
        self.assertEqual(agent.last_decision['heuristicSelection'],[0])

    def test_shadow_never_changes_choice(self):
        agent=JevAgent(HeuristicAgent(catalog=SUITE['catalog']),api_key='test-key',mode='shadow',transport=fake_response)
        self.assertEqual(agent.select(example('save-burn-without-target')),[0])

    def test_cached_request_is_not_rebilled(self):
        agent=JevAgent(api_key='test-key',mode='always',transport=fake_response)
        agent.select(example());agent.select(example())
        self.assertEqual(agent.budget.calls,1)
        self.assertEqual(agent.stats['cacheHits'],1)

    def test_bad_response_explicit_fallback_and_retained_cost(self):
        def bad(payload,key):return {'usage':{'cost':0.003},'answers':{}}
        agent=JevAgent(api_key='test-key',mode='always',transport=bad)
        self.assertEqual(agent.select(example()),[1])
        self.assertTrue(agent.last_decision['fallback'])
        self.assertAlmostEqual(agent.budget.reported_usd,0.003)

    def test_unknown_cost_stops_additional_calls(self):
        def unknown(payload,key):
            response=fake_response(payload,key);response.pop('usage');return response
        agent=JevAgent(api_key='test-key',mode='always',transport=unknown)
        agent.select(example())
        ob=example();ob['current']['turnNumber']=4
        agent.select(ob)
        self.assertEqual(agent.budget.calls,1)
        self.assertTrue(agent.budget.unknown_cost)

    def test_no_key_no_network(self):
        agent=JevAgent(mode='always',transport=lambda *args:self.fail('network'))
        agent.select(example())
        self.assertEqual(agent.last_decision['jev']['reason'],'credential-unavailable')

    def test_call_cap(self):
        agent=JevAgent(api_key='test-key',mode='always',transport=fake_response,budget=Budget(1))
        agent.select(example())
        ob=example();ob['current']['turnNumber']=4
        agent.select(ob)
        self.assertEqual(agent.budget.calls,1)

    def test_compound_never_invented_or_paid(self):
        agent=JevAgent(api_key='test-key',mode='always',transport=lambda *args:self.fail('network'))
        agent.select(example('no-duplicate-attacker'))
        self.assertEqual(agent.last_decision['jev']['reason'],'forced-or-compound-selection')

    def test_score_is_always_offline(self):
        agent=JevAgent(api_key='test-key',mode='always',transport=lambda *args:self.fail('network'))
        self.assertEqual(len(agent.score(example())),2)

    def test_key_absent_from_evidence(self):
        agent=JevAgent(api_key='TEST_SECRET_KEY',mode='always',transport=fake_response)
        agent.select(example())
        self.assertNotIn('TEST_SECRET_KEY',canonical(agent.last_decision))

    def test_nonfinite_probabilities_rejected(self):
        def invalid(payload,key):
            result=fake_response(payload,key);result['answers']['action']['probabilities']['option_0']=float('nan');return result
        agent=JevAgent(api_key='test-key',mode='always',transport=invalid)
        agent.select(example())
        self.assertEqual(agent.last_decision['jev']['status'],'error')

    def test_option_overflow_not_truncated(self):
        ob=example();base=ob['select']['option'][0]
        ob['select']['option']=[dict(base,index=i) for i in range(256)]
        with self.assertRaises(ValueError):payload_for(project(ob))


class TournamentTest(unittest.TestCase):
    def test_replay_javascript_is_parseable(self):
        import re, shutil, subprocess
        if not shutil.which('node'):
            self.skipTest('Node is unavailable for JavaScript syntax validation')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'games.json').write_text('[]')
            render(path,path/'review.html')
            script=re.findall(r'<script>(.*?)</script>',(path/'review.html').read_text(),re.S)[0]
            (path/'review.js').write_text(script)
            result=subprocess.run(['node','--check',str(path/'review.js')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_rotating_player_index_routes_by_identity(self):
        ob=example();ob['select']['playerIndex']=1
        ob['current']['players'][0]['playerIndex']=1
        ob['current']['players'][1]['playerIndex']=0
        self.assertEqual(acting_seat(ob),0)

    def test_unknown_identity_rejected(self):
        ob=example();ob['select']['playerId']='not-a-player'
        with self.assertRaises(ValueError):acting_seat(ob)

    def test_human_runner_enrichment_accepts_parsed_decklists(self):
        class Bridge:
            finished=True
            result={'winner':'Player P0 is the winner'}
            def repository_card_data(self,names):
                self.names=names
                return [{'name':n,'types':['LAND']} for n in names]
            def game_start(self,*args,**kwargs):return {}
        bridge=Bridge();agent=HeuristicAgent()
        play_game(bridge,[agent,agent],[{'name':'Forest','count':60}],
                  [{'name':'Plains','count':60}])
        self.assertEqual(bridge.names,['Forest','Plains'])
        self.assertEqual(set(agent.catalog),{'Forest','Plains'})

    def test_schedule_pairs_both_seats_and_all_decks(self):
        games=schedule(POOL,('heuristic','first'),(1,2))
        self.assertEqual(len(games),36)
        for i in range(0,len(games),2):
            self.assertEqual(games[i]['agents'],games[i+1]['agents'][::-1])
            self.assertEqual(games[i]['decks'],games[i+1]['decks'][::-1])
            self.assertEqual(games[i]['seed'],games[i+1]['seed'])

    def test_turn_limit_not_draw(self):
        self.assertEqual(classify_result({'winner':'Game is a draw','finalState':{'turnNumber':80,'gameEnded':True}},80),('turn-limit',None))

    def test_genuine_draw(self):
        self.assertEqual(classify_result({'winner':'Game is a draw','finalState':{'turnNumber':5,'gameEnded':True}},80),('completed',None))

    def test_missing_winner_not_draw(self):
        self.assertEqual(classify_result({'finalState':{'turnNumber':5}},80),('unresolved-outcome',None))

    def test_exact_player_name(self):
        self.assertEqual(classify_result({'winner':'Player P01 is the winner'},80),('unresolved-outcome',None))
        self.assertEqual(classify_result({'winner':'Player P0 is the winner'},80),('completed',0))

    def test_failures_and_skips_excluded_from_score(self):
        games=[{'status':'completed','agents':['a','b'],'winnerSeat':0},
               {'status':'error','agents':['a','b'],'winnerSeat':None},
               {'status':'skipped','agents':['a','b'],'winnerSeat':None}]
        report=summarize(games)
        self.assertEqual(report['scores']['a']['games'],1)
        self.assertEqual(report['scores']['b']['draws'],0)

    def test_replay_html_escapes_script_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ob=example();ob['select']['option'][0]['label']='</script><script>alert(1)</script>'
            (root/'games.json').write_text(json.dumps([{'id':'g','agents':['a','b'],'decks':['x','y'],'status':'completed','replay':'g.jsonl'}]))
            (root/'g.jsonl').write_text(json.dumps({'observation':ob,'selected':[0],'player':0}))
            render(root,root/'review.html')
            html=(root/'review.html').read_text()
            self.assertNotIn('</script><script>alert',html)
            self.assertIn('\\u003c/script>',html)


class PromotionTest(unittest.TestCase):
    def evidence(self,count=40):
        champion=Config();candidate=propose(champion)[0]['config'];seeds=list(range(1000,1000+count))
        planned=schedule(POOL,('candidate','champion'),seeds)
        games=[dict(s,status='completed',winnerSeat=s['agents'].index('candidate')) for s in planned]
        manifest={'seeds':seeds,'poolHash':digest(POOL),'configs':{'candidate':asdict(candidate),'champion':asdict(champion)},
                  'agents':['candidate','champion'],'completeSchedule':True,'schedule':planned,'codeHash':'code'}
        return candidate,champion,manifest,games

    def gate(self,manifest=None,games=None,count=40,**kwargs):
        candidate,champion,m,g=self.evidence(count)
        return promotion_gate(candidate,champion,SUITE,POOL,[1],manifest or m,games if games is not None else g,
                              expected_code_hash='code',**kwargs)

    def test_sufficient_clean_fixture_evidence_passes_gate(self):
        self.assertTrue(self.gate()['promote'])

    def test_tiny_run_never_promoted(self):
        result=self.gate(count=2)
        self.assertFalse(result['promote'])
        self.assertIn('insufficient-independent-seed-blocks',result['reasons'])

    def test_crashes_block_promotion(self):
        _,_,m,g=self.evidence();g[0]['status']='error'
        self.assertFalse(self.gate(m,g)['promote'])

    def test_missing_games_block_promotion(self):
        _,_,m,g=self.evidence()
        self.assertFalse(self.gate(m,g[:-1])['promote'])

    def test_duplicate_games_block_promotion(self):
        _,_,m,g=self.evidence();g[1]=dict(g[0])
        self.assertFalse(self.gate(m,g)['promote'])

    def test_reused_tuning_seed_blocked(self):
        _,_,m,g=self.evidence();m['seeds'][0]=1
        self.assertIn('holdout-seeds-not-disjoint',self.gate(m,g)['reasons'])

    def test_changed_policy_or_code_blocked(self):
        _,_,m,g=self.evidence();m['codeHash']='changed';m['configs']={}
        result=self.gate(m,g)
        self.assertIn('code-mismatch',result['reasons'])
        self.assertIn('policy-config-mismatch',result['reasons'])

    def test_fifty_percent_is_not_improvement(self):
        _,_,m,g=self.evidence()
        for i,row in enumerate(g):row['winnerSeat']=None
        self.assertFalse(self.gate(m,g)['promote'])

    def test_heldout_selection_not_used_to_propose(self):
        proposals=propose(Config())
        self.assertEqual(len(proposals),10)
        for p in proposals:
            differences=sum(p['config'].weights[k]!=Config().weights[k] for k in Config().weights)
            self.assertEqual(differences,1)

    def test_empty_regression_suite_rejected(self):
        with self.assertRaises(ValueError):regressions(Config(),{'cases':[]})

    def test_invalid_winner_rejected(self):
        _,_,m,g=self.evidence();g[0]['winnerSeat']=2
        self.assertIn('invalid-winner-seat',self.gate(m,g)['reasons'])


if __name__=='__main__':unittest.main()
