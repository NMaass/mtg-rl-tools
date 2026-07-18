import unittest

from mtgo_native_logs.parser import NativeLogParser


class NativeLogTest(unittest.TestCase):
    def test_parse_actions_turn_and_winner(self):
        text = "@P".join([
            "Alice joined the game",
            "Bob joined the game",
            "Turn 1: Alice's turn",
            "Alice plays Forest.",
            "Bob casts Lightning Bolt targeting Alice.",
            "Bob wins the game",
        ])
        result = NativeLogParser().parse_text(text)
        self.assertEqual(["player:1", "player:2"], result.players)
        self.assertEqual(["PLAY_LAND", "CAST_SPELL"],
                         [row.action_type for row in result.actions])
        self.assertEqual("GAME_WIN", result.events[-1].action_type)

    def test_repeated_identical_lines_are_not_deduplicated(self):
        result = NativeLogParser().parse_text(
            "Alice draws a card.\nAlice draws a card.")
        self.assertEqual(2, len(result.actions))

    def test_player_names_can_be_retained_for_private_local_use(self):
        result = NativeLogParser(pseudonymize_players=False).parse_text(
            "Alice joined the game\nBob joined the game\nAlice plays Forest.")
        self.assertEqual(["Alice", "Bob"], result.players)
        self.assertEqual("Alice", result.actions[0].actor)


if __name__ == "__main__":
    unittest.main()
