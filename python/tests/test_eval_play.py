"""Tournament runner, human input parsing, and replay annotation.

Bridge-heavy paths run against a scripted mock bridge -- no XMage startup --
so these stay in the ordinary Python unit suite. The mock mirrors the real
``CabtBridge`` surface the runner touches: ``game_start`` / ``game_select`` /
``game_finish`` plus the ``finished`` and ``result`` attributes.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from magic_cabt.eval.play import (  # noqa: E402
    build_parser,
    play_game,
    run_tournament,
    summarize,
)
from magic_cabt.play.human_vs_agent import (  # noqa: E402
    HumanAgent,
    parse_selection,
    render_prompt,
)
from magic_cabt.replay.annotate import annotate_record, annotate_stream  # noqa: E402
from magic_cabt.agents import make_agent  # noqa: E402


def priority_obs(player_index, option_types):
    options = [{"index": i, "type": t, "label": t, "payload": {}}
               for i, t in enumerate(option_types)]
    return {
        "current": {"turnNumber": 1, "phase": "PRECOMBAT_MAIN",
                    "step": "PRECOMBAT_MAIN",
                    "players": [{"playerIndex": 0, "name": "P0", "life": 20,
                                 "handCount": 7},
                                {"playerIndex": 1, "name": "P1", "life": 20,
                                 "handCount": 7}],
                    "battlefieldSize": 0},
        "select": {"playerIndex": player_index, "type": "PRIORITY",
                   "minCount": 1, "maxCount": 1, "option": options},
    }


class ScriptedBridge(object):
    """A mock bridge that replays a fixed list of observations then ends."""

    def __init__(self, observations, winner="Player P0 is the winner"):
        self._observations = observations
        self._index = 0
        self._winner = winner
        self.finished = False
        self.result = None
        self.selections = []
        self.finish_calls = 0

    def game_start(self, deck0, deck1, player_names=None, seed=None,
                   max_turns=None):
        self._index = 0
        self.finished = False
        self.result = None
        return {"sequence": 0, "observation": self._observations[0]}

    def game_select(self, selection):
        self.selections.append(selection)
        self._index += 1
        if self._index >= len(self._observations):
            self.finished = True
            self.result = {"winner": self._winner, "finalState": {}}
            return {"finished": True, "result": self.result}
        return {"sequence": self._index,
                "observation": self._observations[self._index]}

    def game_finish(self):
        self.finish_calls += 1
        self.finished = True
        return {"ok": True}

    def close(self):
        pass


class PlayGameTest(unittest.TestCase):
    def test_routes_to_acting_seat_and_counts_decisions(self):
        observations = [
            priority_obs(0, ["PASS_PRIORITY", "PLAY_LAND"]),
            priority_obs(1, ["PASS_PRIORITY", "PLAY_LAND"]),
            priority_obs(0, ["PASS_PRIORITY"]),
        ]
        bridge = ScriptedBridge(observations)
        agents = (make_agent("first"), make_agent("first"))
        outcome = play_game(bridge, agents, [], [])

        self.assertTrue(outcome["completed"])
        self.assertEqual(outcome["decisions"], 3)
        self.assertEqual(outcome["winnerSeat"], 0)
        self.assertEqual(outcome["invalidSelections"], 0)
        # both seats take the lowest-index option (PASS_PRIORITY, index 0)
        self.assertEqual(bridge.selections, [[0], [0], [0]])

    def test_winner_seat_maps_seat_one(self):
        bridge = ScriptedBridge([priority_obs(0, ["PASS_PRIORITY"])],
                                winner="Player P1 is the winner")
        outcome = play_game(bridge, (make_agent("first"), make_agent("first")),
                            [], [])
        self.assertEqual(outcome["winnerSeat"], 1)

    def test_draw_when_no_winner(self):
        bridge = ScriptedBridge([priority_obs(0, ["PASS_PRIORITY"])], winner="")
        outcome = play_game(bridge, (make_agent("first"), make_agent("first")),
                            [], [])
        self.assertIsNone(outcome["winnerSeat"])

    def test_illegal_agent_selection_is_counted_and_repaired(self):
        class BadAgent(object):
            name = "bad"

            def select(self, observation):
                return [99]  # out of range

        bridge = ScriptedBridge([priority_obs(0, ["PASS_PRIORITY", "PLAY_LAND"])])
        outcome = play_game(bridge, (BadAgent(), BadAgent()), [], [])
        self.assertEqual(outcome["invalidSelections"], 1)
        self.assertEqual(outcome["invalidBySeat"][0], 1)
        # repaired to a legal selection, so the game still advanced
        self.assertTrue(bridge.selections[0])

    def test_writes_replay_frames(self):
        observations = [priority_obs(0, ["PASS_PRIORITY", "PLAY_LAND"]),
                        priority_obs(0, ["PASS_PRIORITY"])]
        bridge = ScriptedBridge(observations)
        frames = []

        class Writer(object):
            game_id = "game-0000"

            def write_frame(self, frame):
                frames.append(frame)

        play_game(bridge, (make_agent("first"), make_agent("first")),
                  [], [], record_writer=Writer())
        # one frame per decision + a trailing result frame
        self.assertEqual(len(frames), 3)
        self.assertIn("selected", frames[0])
        self.assertEqual(frames[0]["gameId"], "game-0000")
        self.assertIn("result", frames[-1])


class RunTournamentTest(unittest.TestCase):
    def test_runs_multiple_games_and_writes_outputs(self):
        observations = [priority_obs(0, ["PASS_PRIORITY", "PLAY_LAND"]),
                        priority_obs(0, ["PASS_PRIORITY"])]
        bridge = ScriptedBridge(observations)
        out_dir = tempfile.mkdtemp()
        summary = run_tournament(
            ("first", "random"), [], [], games=3, seed=1,
            bridge=bridge, out_dir=out_dir)

        self.assertEqual(summary["gamesAttempted"], 3)
        self.assertEqual(summary["gamesCompleted"], 3)
        self.assertEqual(summary["agents"], {"seat0": "first",
                                             "seat1": "random"})
        self.assertTrue(os.path.exists(os.path.join(out_dir, "summary.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "game-0000.jsonl")))
        # replay is readable as self-play frames
        with open(os.path.join(out_dir, "game-0000.jsonl")) as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        self.assertIn("result", lines[-1])


class SummarizeTest(unittest.TestCase):
    def test_aggregates_wins_losses_draws_and_invalids(self):
        outcomes = [
            {"completed": True, "winnerSeat": 0, "decisions": 10,
             "invalidBySeat": [1, 0], "failClosed": False},
            {"completed": True, "winnerSeat": 1, "decisions": 20,
             "invalidBySeat": [0, 2], "failClosed": False},
            {"completed": True, "winnerSeat": None, "decisions": 30,
             "invalidBySeat": [0, 0], "failClosed": False},
            {"completed": False, "winnerSeat": None, "decisions": 0,
             "invalidBySeat": [0, 0], "failClosed": True,
             "error": "boom"},
        ]
        summary = summarize(outcomes, ("random", "first"))
        self.assertEqual(summary["gamesAttempted"], 4)
        self.assertEqual(summary["gamesCompleted"], 3)
        self.assertEqual(summary["crashes"], 1)
        self.assertEqual(summary["failClosed"], 1)
        self.assertEqual(summary["winsBySeat"], {"0": 1, "1": 1})
        self.assertEqual(summary["lossesBySeat"], {"0": 1, "1": 1})
        self.assertEqual(summary["draws"], 1)
        self.assertEqual(summary["totalDecisions"], 60)
        self.assertEqual(summary["averageDecisionsPerGame"], 20.0)
        self.assertEqual(summary["invalidSelections"],
                         {"total": 3, "seat0": 1, "seat1": 2})
        self.assertEqual(summary["errors"], ["boom"])


class ArgParsingTest(unittest.TestCase):
    def test_parses_agent_specs_and_options(self):
        args = build_parser().parse_args([
            "--agent0", "random", "--agent1", "first",
            "--games", "20", "--seed", "1",
            "--deck0", "a.txt", "--deck1", "b.txt",
            "--out", "target/eval/x", "--fail-fast"])
        self.assertEqual(args.agent0, "random")
        self.assertEqual(args.agent1, "first")
        self.assertEqual(args.games, 20)
        self.assertEqual(args.seed, 1)
        self.assertTrue(args.fail_fast)


class HumanInputTest(unittest.TestCase):
    def setUp(self):
        self.single = {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                       "option": [{"index": 0, "type": "PASS_PRIORITY"},
                                  {"index": 1, "type": "PLAY_LAND"}]}
        self.multi = {"type": "MODE", "minCount": 1, "maxCount": 2,
                      "option": [{"index": 0, "type": "PROMPT_MODE"},
                                 {"index": 1, "type": "PROMPT_MODE"},
                                 {"index": 2, "type": "PROMPT_MODE"}]}
        self.optional = {"type": "DECLARE_ATTACKERS", "minCount": 0,
                         "maxCount": 0,
                         "option": [{"index": 0, "type": "PROMPT_ATTACKER"}]}

    def test_single_index(self):
        self.assertEqual(parse_selection("1", self.single), [1])

    def test_multi_index_space_and_comma(self):
        self.assertEqual(parse_selection("0 2", self.multi), [0, 2])
        self.assertEqual(parse_selection("0,1", self.multi), [0, 1])

    def test_blank_allowed_only_when_optional(self):
        self.assertEqual(parse_selection("", self.optional), [])
        with self.assertRaises(ValueError):
            parse_selection("", self.single)

    def test_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            parse_selection("land", self.single)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            parse_selection("5", self.single)

    def test_too_many_rejected(self):
        with self.assertRaises(ValueError):
            parse_selection("0 1 2", self.multi)

    def test_human_agent_reprompts_until_legal(self):
        inputs = iter(["nope", "9", "1"])
        outputs = []
        agent = HumanAgent(input_fn=lambda prompt: next(inputs),
                           output_fn=outputs.append)
        observation = priority_obs(0, ["PASS_PRIORITY", "PLAY_LAND"])
        self.assertEqual(agent.select(observation), [1])
        # rendered the prompt once, complained about the two bad inputs
        self.assertTrue(any("invalid input" in str(o) for o in outputs))

    def test_render_prompt_lists_options(self):
        text = render_prompt(priority_obs(0, ["PASS_PRIORITY", "PLAY_LAND"]))
        self.assertIn("PLAY_LAND", text)
        self.assertIn("[1]", text)


class AnnotateTest(unittest.TestCase):
    def _record(self, selected):
        return {
            "gameId": "game-1",
            "sequenceNumber": 42,
            "select": {"type": "PRIORITY", "minCount": 1, "maxCount": 1,
                       "option": [
                           {"index": 0, "type": "PASS_PRIORITY",
                            "label": "Pass priority"},
                           {"index": 1, "type": "PLAY_LAND",
                            "label": "Play Mountain"},
                           {"index": 2, "type": "CAST_SPELL",
                            "label": "Cast Bears"}]},
            "selectedIndices": selected,
        }

    def test_topk_and_chosen_rank(self):
        scorer = make_agent("first")
        annotation = annotate_record(self._record([1]), scorer, top_k=2)
        self.assertEqual(annotation["gameId"], "game-1")
        self.assertEqual(annotation["sequenceNumber"], 42)
        self.assertEqual(annotation["policy"], "first")
        self.assertEqual(len(annotation["topK"]), 2)
        # first ranks option 0 first; chosen index 1 lands at rank 2
        self.assertEqual(annotation["topK"][0]["index"], 0)
        self.assertEqual(annotation["topK"][0]["label"], "Pass priority")
        self.assertEqual(annotation["chosenRank"], 2)
        self.assertEqual(annotation["chosenScore"], annotation["topK"][1]["score"])

    def test_chosen_rank_reflects_top_choice(self):
        scorer = make_agent("first")
        annotation = annotate_record(self._record([0]), scorer, top_k=5)
        # first option (index 0) ranks first
        self.assertEqual(annotation["chosenRank"], 1)

    def test_missing_choice_yields_null_rank(self):
        scorer = make_agent("first")
        annotation = annotate_record(self._record([]), scorer)
        self.assertIsNone(annotation["chosenRank"])
        self.assertIsNone(annotation["chosenScore"])

    def test_annotate_stream_skips_optionless_records(self):
        scorer = make_agent("first")
        records = [self._record([1]),
                   {"gameId": "g", "sequenceNumber": 1,
                    "select": {"type": "PRIORITY", "option": []},
                    "selectedIndices": []}]
        annotations = list(annotate_stream(records, scorer))
        self.assertEqual(len(annotations), 1)


if __name__ == "__main__":
    unittest.main()
