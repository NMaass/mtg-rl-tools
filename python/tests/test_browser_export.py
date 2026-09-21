import copy
import json
import unittest
from magic_cabt.browser_export import parse_export, project_snapshot, project_decision
from test_arena_mirror import SessionAutoOpenTest

class BrowserExportTest(unittest.TestCase):
    def test_original_normalizer_and_tracker_create_browser_replay(self):
        replays=json.loads(parse_export(SessionAutoOpenTest()._log_text(),'arena'))
        self.assertTrue(replays)
        self.assertTrue(replays[0]['frames'])
        self.assertTrue(any(f['decisions'] for f in replays[0]['frames']))

    def test_private_hands_never_enter_other_perspectives(self):
        snapshot={'localSeat':1,'players':[{'seat':1,'life':20},{'seat':2,'life':20}],
                  'zones':{'hands':{'1':[{'instanceId':3,'name':'Island'}],
                                    '2':[{'instanceId':4,'name':'SECRET'}]}}}
        view,_=project_snapshot(snapshot)
        self.assertNotIn('SECRET',json.dumps(view))
        self.assertIn('Island',json.dumps(view))

    def test_face_down_identity_and_cache_are_not_exposed(self):
        snapshot={'localSeat':1,'players':[{'seat':1},{'seat':2}],
                  'zones':{'battlefield':[{'instanceId':5,'grpId':9,'name':'SECRET','faceDown':True}]}}
        view,_=project_snapshot(snapshot,{'arena':{'9':{'name':'SECRET','oracleText':'PRIVATE'}}})
        self.assertNotIn('SECRET',json.dumps(view))
        self.assertNotIn('PRIVATE',json.dumps(view))

    def test_duplicate_card_instances_remain_distinguishable(self):
        snapshot={'localSeat':1,'players':[{'seat':1},{'seat':2}],
                  'zones':{'battlefield':[{'instanceId':5,'name':'Mountain'},{'instanceId':6,'name':'Mountain'}]}}
        view,aliases=project_snapshot(snapshot)
        self.assertNotEqual(aliases['5'],aliases['6'])
        self.assertEqual(len(view['cards']),2)

    def test_mtgo_text_never_claims_priority_or_hidden_hands(self):
        text='Alice joined the game.\nBob joined the game.\nAlice begins the game with seven cards in hand.\nBob begins the game with seven cards in hand.\nTurn 1: Alice\nAlice plays Mountain.\n'
        replay=json.loads(parse_export(text,'mtgo'))[0]
        self.assertEqual(replay['source'],'mtgo')
        for f in replay['frames']:
            self.assertEqual(f['decisions'],{})
            self.assertIsNone(f['views']['public']['priority'])
            self.assertTrue(all(p['hand']==[] for p in f['views']['public']['players']))

    def test_binary_mtgo_fails_explicitly(self):
        with self.assertRaises(ValueError):parse_export('\x00invalid','mtgo')

if __name__=='__main__':unittest.main()
