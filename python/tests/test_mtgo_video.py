"""Tests for the MTGO video ingestion pipeline.

These cover the parts that do not need ffmpeg, tesseract, or XMage: OCR line
assembly, scrolling-window reconstruction, the MTGO log grammar, game
splitting, and board simulation. The XMage agreement check itself lives in
magic_cabt.mtgo_video.verify and is exercised by the mirror verification
script (it needs a built Mage.Client classpath).
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from magic_cabt.mtgo_video.catalog import (
    CardCatalog,
    _informative,
    fold,
    is_creature,
    is_permanent,
)
from magic_cabt.mtgo_video.games import (
    discover_players,
    game_is_complete,
    split_games,
)
from magic_cabt.mtgo_video.ocr import (
    lines_to_entries,
    strip_junk,
    strip_timestamp,
    TIMESTAMP_RE,
)
from magic_cabt.mtgo_video.parse import parse_entry, parse_log
from magic_cabt.mtgo_video.reconstruct import LogReconstructor, normalize
from magic_cabt.mtgo_video.regions import Region
from magic_cabt.mtgo_video.state import GameSimulator
from magic_cabt.mtgo_video.verify import compare_state


class OcrAssemblyTest(unittest.TestCase):
    def test_wrapped_lines_join_into_one_entry(self):
        lines = [
            "7:07 AM: BuzzCaldera mills Sleep of the Dead and Cryptic",
            "Serpent.",
            "7:07 AM: BuzzCaldera draws a card with Mental Note.",
        ]
        entries = lines_to_entries(lines)
        self.assertEqual(entries, [
            "7:07 AM: BuzzCaldera mills Sleep of the Dead and Cryptic Serpent.",
            "7:07 AM: BuzzCaldera draws a card with Mental Note.",
        ])

    def test_orphan_continuation_at_top_of_pane_is_dropped(self):
        # The pane scrolls up, so a leading continuation line's parent entry
        # was already captured in full by an earlier frame.
        lines = ["Aberration.", "7:07 AM: BuzzCaldera draws a card."]
        self.assertEqual(lines_to_entries(lines),
                         ["7:07 AM: BuzzCaldera draws a card."])

    def test_glued_lines_split_on_inline_timestamp(self):
        lines = ["7:04 AM: golubtsov draws a card. 7:04 AM: golubtsov has conceded."]
        self.assertEqual(lines_to_entries(lines), [
            "7:04 AM: golubtsov draws a card.",
            "7:04 AM: golubtsov has conceded.",
        ])

    def test_mangled_timestamps_still_start_an_entry(self):
        for stamp in ("7:07 AM:", "/:06 AN:", "7:04 AWM:", "7:01 4M:", "7:06 Ali:"):
            with self.subTest(stamp=stamp):
                self.assertTrue(TIMESTAMP_RE.match(stamp + " BuzzCaldera draws a card."))

    def test_strip_junk_removes_scrollbar_glyphs_but_keeps_content(self):
        self.assertEqual(strip_junk("golubtsov draws a card. VU"),
                         "golubtsov draws a card.")
        self.assertEqual(strip_junk("Turn 3: BuzzCaldera"), "Turn 3: BuzzCaldera")

    def test_strip_timestamp(self):
        self.assertEqual(strip_timestamp("7:07 AM: BuzzCaldera plays Island."),
                         "BuzzCaldera plays Island.")


class RegionTest(unittest.TestCase):
    def test_parse_and_ffmpeg_crop(self):
        region = Region.parse("345x445+1544+88")
        self.assertEqual((region.width, region.height, region.x, region.y),
                         (345, 445, 1544, 88))
        self.assertEqual(region.ffmpeg_crop(), "crop=345:445:1544:88")


class ReconstructTest(unittest.TestCase):
    def test_overlapping_windows_merge_without_duplication(self):
        rec = LogReconstructor()
        rec.feed(["7:00 AM: a plays Island.", "7:00 AM: a casts Brainstorm."])
        rec.feed(["7:00 AM: a casts Brainstorm.", "7:00 AM: a draws a card."])
        self.assertEqual(rec.entries, [
            "7:00 AM: a plays Island.",
            "7:00 AM: a casts Brainstorm.",
            "7:00 AM: a draws a card.",
        ])

    def test_ocr_variants_collapse_and_keep_longest_reading(self):
        rec = LogReconstructor()
        rec.feed(["7:00 AM: a mills Cryptic"])
        rec.feed(["7:00 AM: a mills Cryptic Serpent and Island."])
        self.assertEqual(rec.entries, ["7:00 AM: a mills Cryptic Serpent and Island."])

    def test_repeated_identical_frames_add_nothing(self):
        rec = LogReconstructor()
        frame = ["7:00 AM: a plays Island.", "7:00 AM: a draws a card."]
        rec.feed(frame)
        for _ in range(5):
            self.assertEqual(rec.feed(list(frame)), 0)
        self.assertEqual(len(rec.entries), 2)

    def test_new_entries_are_not_inserted_mid_log(self):
        # A late frame that falsely aligns against old entries must not
        # splice its content into the middle: the log is append-only.
        rec = LogReconstructor()
        rec.feed(["7:00 AM: a plays Island.", "7:00 AM: a draws a card.",
                  "7:00 AM: a casts Brainstorm."])
        rec.feed(["7:00 AM: a plays Island.", "7:05 AM: b joined the game.",
                  "7:00 AM: a casts Brainstorm."])
        self.assertNotIn("7:05 AM: b joined the game.", rec.entries[:-1])

    def test_normalize_ignores_timestamp_and_punctuation(self):
        self.assertEqual(normalize("7:07 AM: a plays Island."),
                         normalize("/:07 AN: a plays Island"))

    def test_entries_keep_the_earliest_frame_they_appeared_in(self):
        rec = LogReconstructor()
        rec.feed(["7:00 AM: a plays Island."], timestamp=10.0)
        rec.feed(["7:00 AM: a plays Island.", "7:00 AM: a draws a card."],
                 timestamp=11.0)
        # Re-seeing an entry in a later frame must not move its timestamp.
        rec.feed(["7:00 AM: a plays Island."], timestamp=12.0)
        self.assertEqual(rec.times, [10.0, 11.0])

    def test_majority_reading_wins_across_repeated_sightings(self):
        # A line sits on screen for many frames and OCR does not fail the
        # same way each time, so the majority reading beats any single one.
        rec = LogReconstructor()
        for text in ["7:00 AM: a casts Mental Note.",
                     "7:00 AM: a casts Mentol Note.",
                     "7:00 AM: a casts Mental Note.",
                     "7:00 AM: a casts Mental Note."]:
            rec.feed([text])
        self.assertEqual(rec.entries, ["7:00 AM: a casts Mental Note."])

    def test_a_misread_duplicate_is_folded_back_together(self):
        rec = LogReconstructor()
        rec.feed(["7:00 AM: a plays Island.", "7:00 AM: a casts Brainstorm."])
        # A noisy re-read that failed to align gets appended as a duplicate.
        rec.feed(["7:00 AM: a casts Brainstonn."])
        self.assertEqual(len(rec.entries), 2)

    def test_a_genuinely_repeated_line_is_not_collapsed(self):
        # MTGO really does log the same sentence twice in a row. The two are
        # on screen together, which is what tells them apart from a misread.
        rec = LogReconstructor()
        both = ["7:03 AM: a casts Tolarian Terror.",
                "7:03 AM: a casts Tolarian Terror."]
        rec.feed(both)
        rec.feed(both)
        self.assertEqual(len(rec.entries), 2)

    def test_longer_reading_wins_but_earliest_time_is_kept(self):
        rec = LogReconstructor()
        rec.feed(["7:00 AM: a mills Cryptic"], timestamp=5.0)
        rec.feed(["7:00 AM: a mills Cryptic Serpent and Island."], timestamp=6.0)
        self.assertEqual(rec.entries, ["7:00 AM: a mills Cryptic Serpent and Island."])
        self.assertEqual(rec.times, [5.0])


class ParseTest(unittest.TestCase):
    def test_core_grammar(self):
        cases = [
            ("7:00 AM: Turn 3: BuzzCaldera", {"type": "TURN", "turn": 3,
                                              "player": "BuzzCaldera"}),
            ("7:00 AM: BuzzCaldera plays Island.", {"type": "PLAY_LAND",
                                                    "card": "Island"}),
            ("7:00 AM: BuzzCaldera casts Thought Scour targeting BuzzCaldera.",
             {"type": "CAST", "card": "Thought Scour",
              "targets": ["BuzzCaldera"]}),
            ("7:00 AM: BuzzCaldera mills Cryptic Serpent and Island.",
             {"type": "MILL", "cards": ["Cryptic Serpent", "Island"]}),
            ("7:00 AM: Delver of Secrets transforms into Insectile Aberration.",
             {"type": "TRANSFORM", "card": "Delver of Secrets",
              "into": "Insectile Aberration"}),
            ("7:00 AM: BuzzCaldera counters Candy Trail with Annul.",
             {"type": "COUNTERS_WITH", "card": "Candy Trail",
              "counter": "Annul"}),
            ("7:00 AM: golubtsov has conceded from the game.",
             {"type": "CONCEDE", "player": "golubtsov"}),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                event = parse_entry(text)
                for key, value in expected.items():
                    self.assertEqual(event[key], value)

    def test_alternate_cost_is_not_part_of_the_card_name(self):
        event = parse_entry("7:03 AM: a casts Boulderbranch Golem with Prototype.")
        self.assertEqual(event["card"], "Boulderbranch Golem")
        self.assertEqual(event["alternate"], "Prototype")

    def test_ocr_mangled_turn_number_is_recovered(self):
        self.assertEqual(parse_entry("7:03 AM: Turn 5S: golubtsov")["turn"], 5)

    def test_draw_count_comes_from_the_noun_not_the_article(self):
        # The article is one glyph and misreads often ("draws 3 card"); the
        # singular noun is the reliable signal that it was one card.
        cases = [
            ("7:00 AM: a draws a card.", 1),
            ("7:00 AM: a draws 3 card with Thought Scour.", 1),
            ("7:00 AM: a draws three cards with Brainstorm.", 3),
            ("7:00 AM: a draws 3 cards.", 3),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_entry(text)["count"], expected)

    def test_a_misread_clock_separator_still_delimits_entries(self):
        # OCR reads the colon in "7:01 AM" as a hyphen often enough that it
        # would otherwise glue two log lines into one unparsable entry.
        self.assertEqual(
            lines_to_entries(["7:01 AM: a draws a card. 7-01 AM: a plays Bojuka Bog."]),
            ["7:01 AM: a draws a card.", "7-01 AM: a plays Bojuka Bog."])

    def test_a_misread_verb_is_repaired_only_if_the_result_parses(self):
        repaired = parse_entry("7:02 AM: a eyeles Lorien Revealed.")
        self.assertEqual(repaired["type"], "CYCLE")
        self.assertTrue(repaired["repaired"])
        # The original text is kept, so the repair stays auditable.
        self.assertIn("eyeles", repaired["text"])

    def test_verb_repair_does_not_invent_events_from_noise(self):
        for junk in ("7:00 AM: total gibberish zzz qqq",
                     "7:00 AM: xxxxx yyyyy zzzzz"):
            with self.subTest(junk=junk):
                self.assertEqual(parse_entry(junk)["type"], "UNPARSED")

    def test_a_mangled_clock_still_delimits_and_parses(self):
        # Every one of these separators and AM spellings came out of a real
        # capture; each would otherwise glue two log lines into one.
        glued = ("7:00 AM: Turn 1: golubtsov 7,00 AM: golubtsov skips their "
                 "draw step. 7203 AM: golubtsov draws a card. "
                 "7:01 Alvi: golubtsov plays Forest.")
        entries = lines_to_entries([glued])
        self.assertEqual([parse_entry(e)["type"] for e in entries],
                         ["TURN", "SKIP_DRAW", "DRAW", "PLAY_LAND"])

    def test_unrecognized_entries_are_reported_not_dropped(self):
        event = parse_entry("7:00 AM: something entirely new happens somehow")
        self.assertEqual(event["type"], "UNPARSED")
        self.assertIn("something entirely new", event["text"])


class LayoutTest(unittest.TestCase):
    """Locating the UI without depending on the capture resolution."""

    def frame(self, width=1920, height=1080, letterbox=0):
        """A synthetic MTGO-ish frame: dark board, bright log pane, scrollbar."""
        from PIL import Image, ImageDraw

        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        scale = width / 1920.0
        inner_top = letterbox
        inner_height = height - 2 * letterbox
        # board
        draw.rectangle([0, inner_top, width - 1, inner_top + inner_height - 1],
                       fill=80)
        # log pane: white text column plus a darker scrollbar at its right
        pane_x0 = int(1537 * scale)
        pane_x1 = int(1892 * scale)
        bar_x1 = int(1906 * scale)
        pane_y0 = inner_top + int(62 * scale)
        pane_y1 = inner_top + int(538 * scale)
        draw.rectangle([pane_x0, pane_y0, pane_x1, pane_y1], fill=254)
        draw.rectangle([pane_x1 + 1, pane_y0, bar_x1, pane_y1], fill=162)
        # the bright border column drawn outside the scrollbar
        draw.rectangle([bar_x1 + 1, pane_y0, bar_x1 + 3, pane_y1], fill=254)
        return image

    def test_pane_is_found_and_excludes_the_scrollbar(self):
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(self.frame())
        self.assertTrue(layout.detected)
        right = layout.log_pane.x + layout.log_pane.width
        # The text column ends before the scrollbar at x=1893, not after the
        # bright border column that sits outside it.
        self.assertLess(right, 1893)
        self.assertGreater(right, 1860)

    def test_detection_scales_with_capture_resolution(self):
        from magic_cabt.mtgo_video.layout import build_layout

        panes = {}
        for width, height in ((1280, 720), (1920, 1080), (2560, 1440)):
            layout = build_layout(self.frame(width, height))
            self.assertTrue(layout.detected, "%dx%d" % (width, height))
            panes[width] = layout.log_pane
        # Widths should track the resolution ratio within rounding.
        self.assertAlmostEqual(panes[1280].width / panes[1920].width,
                               1280 / 1920, delta=0.03)
        self.assertAlmostEqual(panes[2560].width / panes[1920].width,
                               2560 / 1920, delta=0.03)

    def test_letterboxing_is_trimmed_before_locating_anything(self):
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(self.frame(1920, 1200, letterbox=60))
        self.assertEqual(layout.content.y, 60)
        self.assertEqual(layout.content.height, 1080)
        self.assertTrue(layout.detected)

    def test_falls_back_to_proportional_regions_when_nothing_is_found(self):
        from PIL import Image
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(Image.new("L", (1920, 1080), 80))
        self.assertFalse(layout.detected)
        self.assertGreater(layout.log_pane.width, 0)

    def test_consensus_uses_the_narrowest_credible_right_edge(self):
        # A frame whose log has not overflowed yet shows no scrollbar, so its
        # pane looks wider. Taking that would glue scrollbar glyphs onto every
        # wrapped line in the frames that do have one.
        import os
        import tempfile

        from magic_cabt.mtgo_video.layout import detect_layout_from_frames

        work = tempfile.mkdtemp(prefix="layout_test_")
        try:
            paths = []
            for index in range(3):
                image = self.frame()
                if index == 2:  # no scrollbar drawn: pane reads wider
                    from PIL import ImageDraw
                    ImageDraw.Draw(image).rectangle(
                        [1893, 62, 1906, 538], fill=254)
                path = os.path.join(work, "f%d.png" % index)
                image.save(path)
                paths.append(path)
            layout = detect_layout_from_frames(paths)
            self.assertLess(layout.log_pane.x + layout.log_pane.width, 1893)
            self.assertEqual(layout.samples, 3)
        finally:
            import shutil
            shutil.rmtree(work, ignore_errors=True)


class CatalogTest(unittest.TestCase):
    """Offline card-name resolution: the pipeline never guesses a card."""

    CARDS = {
        "Bojuka Bog": {"name": "Bojuka Bog", "type_line": "Land"},
        "Lórien Revealed": {"name": "Lórien Revealed", "type_line": "Sorcery"},
        "Mental Note": {"name": "Mental Note", "type_line": "Instant"},
        "Tolarian Terror": {"name": "Tolarian Terror",
                            "type_line": "Creature — Serpent",
                            "power": "5", "toughness": "5"},
        "Delver of Secrets": {"name": "Delver of Secrets",
                              "type_line": "Creature — Human Wizard",
                              "power": "1", "toughness": "1"},
        "Island": {"name": "Island", "type_line": "Basic Land — Island"},
        "Counterspell": {"name": "Counterspell", "type_line": "Instant"},
    }

    def catalog(self, vocabulary=()):
        return CardCatalog.from_cards(dict(self.CARDS), vocabulary=vocabulary)

    def test_exact_and_accent_insensitive_lookup(self):
        catalog = self.catalog()
        self.assertEqual(catalog.resolve("Island"), "Island")
        self.assertEqual(catalog.resolve("Lorien Revealed"), "Lórien Revealed")
        self.assertEqual(catalog.resolve("LÓRIEN REVEALED"), "Lórien Revealed")

    def test_ocr_typo_resolves_against_the_run_vocabulary(self):
        catalog = self.catalog(vocabulary=["Bojuka Bog", "Mental Note"])
        self.assertEqual(catalog.resolve("Bojuko Bog"), "Bojuka Bog")
        self.assertEqual(catalog.resolve("Mentol Note"), "Mental Note")

    def test_player_names_are_rejected_not_matched_to_a_card(self):
        catalog = self.catalog(vocabulary=list(self.CARDS))
        for junk in ("Buz2Caldera", "golubtsovy", "BuzzCaldera"):
            with self.subTest(junk=junk):
                self.assertIsNone(catalog.resolve(junk))

    def test_resolution_is_memoized_so_a_string_always_maps_the_same_way(self):
        catalog = self.catalog(vocabulary=["Bojuka Bog"])
        first = catalog.resolve("Bojuko Bog")
        catalog.add_to_vocabulary("Counterspell")
        self.assertEqual(catalog.resolve("Bojuko Bog"), first)

    def test_seed_from_log_takes_only_exactly_spelled_names(self):
        catalog = self.catalog()
        catalog.seed_from_log(["Bojuka Bog", "Bojuko Bog", "Buz2Caldera"])
        self.assertEqual(catalog.vocabulary, ["Bojuka Bog"])

    def test_unresolved_names_are_counted_for_reporting(self):
        catalog = self.catalog()
        catalog.resolve("Buz2Caldera")
        self.assertIn("Buz2Caldera", catalog.unresolved)

    def test_lookup_returns_metadata_for_the_simulator(self):
        catalog = self.catalog()
        info = catalog.lookup("Tolarian Terror")
        self.assertTrue(is_permanent(info))
        self.assertTrue(is_creature(info))
        self.assertFalse(is_permanent(catalog.lookup("Counterspell")))

    def test_fold_normalizes_case_accents_and_punctuation(self):
        self.assertEqual(fold("Urza's Power-Plant"), "urza s power plant")
        self.assertEqual(fold("Lórien Revealed"), fold("Lorien Revealed"))

    def test_placeholder_entries_never_displace_a_real_card(self):
        # Scryfall art-series cards reuse real names with type_line "Card".
        self.assertFalse(_informative({"type_line": "Card"}))
        self.assertTrue(_informative({"type_line": "Creature — Serpent"}))


class VideoTimeTest(unittest.TestCase):
    def test_parse_log_attaches_video_timestamps(self):
        events = parse_log(["7:00 AM: a plays Island."], [12.25])
        self.assertEqual(events[0]["videoTime"], 12.25)


class DiscoverPlayersTest(unittest.TestCase):
    def test_seat_order_comes_from_pregame_lines(self):
        events = parse_log([
            "6:59 AM: golubtsov rolled a 3.",
            "6:59 AM: BuzzCaldera rolled a 4.",
            "6:59 AM: Turn 1: BuzzCaldera",
        ])
        self.assertEqual(discover_players(events), ["golubtsov", "BuzzCaldera"])

    def test_ocr_variants_do_not_create_extra_players(self):
        events = parse_log([
            "6:59 AM: golubtsov joined the game.",
            "6:59 AM: BuzzCaldera joined the game.",
            "6:59 AM: golubtsovy draws a card.",
            "6:59 AM: Buz2zCaldera draws a card.",
        ])
        self.assertEqual(discover_players(events), ["golubtsov", "BuzzCaldera"])

    def test_falls_back_to_the_whole_game_when_pregame_is_missing(self):
        events = parse_log([
            "7:02 AM: BuzzCaldera plays Island.",
            "7:02 AM: golubtsov draws a card.",
        ])
        self.assertEqual(sorted(discover_players(events)),
                         ["BuzzCaldera", "golubtsov"])


class GameSplitTest(unittest.TestCase):
    def test_match_splits_at_each_new_game(self):
        events = parse_log([
            "6:59 AM: a joined the game.",
            "6:59 AM: Turn 1: a",
            "7:04 AM: a wins the game.",
            "7:05 AM: a joined the game.",
            "7:06 AM: Turn 1: b",
        ])
        games = split_games(events)
        self.assertEqual(len(games), 2)
        self.assertTrue(game_is_complete(games[0]))
        self.assertFalse(game_is_complete(games[1]))


class SimulatorTest(unittest.TestCase):
    def _sim(self, texts, **kwargs):
        sim = GameSimulator(deck_size=60, **kwargs)
        states = []
        for event in parse_log(texts):
            states.extend(sim.apply(event))
        return sim, states

    def test_lands_and_spells_reach_the_right_zones(self):
        info = {"Island": {"type_line": "Basic Land — Island"},
                "Brainstorm": {"type_line": "Instant"}}
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: Turn 1: a",
             "7:00 AM: a plays Island.",
             "7:00 AM: a casts Brainstorm."],
            card_info=lambda name: info.get(name),
        )
        zones = states[-1]["zones"]
        self.assertEqual([o["name"] for o in zones["battlefield"]], ["Island"])
        self.assertEqual([o["name"] for o in zones["graveyards"]["1"]], ["Brainstorm"])

    def test_hand_and_library_counts_track_draws(self):
        sim, states = self._sim([
            "7:00 AM: a begins the game with seven cards in hand.",
            "7:00 AM: a draws a card.",
        ])
        player = states[-1]["players"][0]
        self.assertEqual((player["handCount"], player["libraryCount"]), (8, 52))

    def test_damage_to_a_player_moves_life(self):
        sim, states = self._sim([
            "7:00 AM: a begins the game with seven cards in hand.",
            "7:00 AM: b begins the game with seven cards in hand.",
            "7:00 AM: Insectile Aberration deals 3 damage to b.",
        ])
        lives = {p["name"]: p["life"] for p in states[-1]["players"]}
        self.assertEqual(lives, {"a": 20, "b": 17})

    def test_attackers_are_tapped(self):
        info = {"Insectile Aberration": {"type_line": "Creature — Human Insect",
                                         "power": "3", "toughness": "2"}}
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand.",
             "7:00 AM: Turn 3: a",
             "7:00 AM: a casts Insectile Aberration.",
             "7:00 AM: b is being attacked by Insectile Aberration."],
            card_info=lambda name: info.get(name),
        )
        creature = states[-1]["zones"]["battlefield"][0]
        self.assertTrue(creature["tapped"])
        self.assertEqual((creature["power"], creature["toughness"]), (3, 2))
        self.assertEqual(states[-1]["step"], "Step_DeclareAttack")

    def test_unblocked_attackers_deal_their_damage(self):
        # MTGO logs no combat-damage line for an unblocked attack; the life
        # loss has to come from the attack declaration itself.
        info = {"Tolarian Terror": {"type_line": "Creature — Serpent",
                                    "power": "5", "toughness": "5"}}
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand.",
             "7:00 AM: Turn 5: a",
             "7:00 AM: a casts Tolarian Terror.",
             "7:00 AM: a casts Tolarian Terror.",
             "7:00 AM: b is being attacked by Tolarian Terror and Tolarian Terror.",
             "7:00 AM: Turn 5: b"],
            card_info=lambda name: info.get(name),
        )
        lives = {p["name"]: p["life"] for p in states[-1]["players"]}
        self.assertEqual(lives, {"a": 20, "b": 10})

    def test_combat_damage_is_its_own_state_stamped_to_the_attack(self):
        # Declaring attackers does not itself change life; the damage lands
        # once combat resolves. It gets its own snapshot so the life change
        # is attributed to the attack's moment in the video rather than to
        # whenever the next line happened to be logged.
        info = {"Tolarian Terror": {"type_line": "Creature — Serpent",
                                    "power": "5", "toughness": "5"}}
        sim = GameSimulator(deck_size=60, card_info=lambda n: info.get(n))
        texts = ["7:00 AM: a begins the game with seven cards in hand.",
                 "7:00 AM: b begins the game with seven cards in hand.",
                 "7:00 AM: Turn 5: a",
                 "7:00 AM: a casts Tolarian Terror.",
                 "7:00 AM: b is being attacked by Tolarian Terror.",
                 "7:00 AM: a draws a card."]
        events = parse_log(texts, [1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
        states = []
        for event in events:
            states.extend(sim.apply(event))

        at_attack = states[-3]
        damage = states[-2]
        self.assertEqual(
            {p["name"]: p["life"] for p in at_attack["players"]}["b"], 20)
        self.assertEqual(
            {p["name"]: p["life"] for p in damage["players"]}["b"], 15)
        self.assertEqual(damage["sourceEvent"]["type"], "COMBAT_DAMAGE")
        # Stamped to the attack (100.0), not to the next logged line (200.0).
        self.assertEqual(damage["videoTime"], 100.0)

    def test_a_blocked_attacker_deals_no_damage_to_the_player(self):
        info = {"Tolarian Terror": {"type_line": "Creature — Serpent",
                                    "power": "5", "toughness": "5"},
                "Generous Ent": {"type_line": "Creature — Treefolk",
                                 "power": "5", "toughness": "7"}}
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand.",
             "7:00 AM: Turn 5: a",
             "7:00 AM: a casts Tolarian Terror.",
             "7:00 AM: b is being attacked by Tolarian Terror.",
             "7:00 AM: b blocks Tolarian Terror with Generous Ent.",
             "7:00 AM: Turn 5: b"],
            card_info=lambda name: info.get(name),
        )
        lives = {p["name"]: p["life"] for p in states[-1]["players"]}
        self.assertEqual(lives["b"], 20)

    def test_an_attacker_with_unknown_power_is_flagged_not_guessed(self):
        sim, states = self._sim([
            "7:00 AM: a begins the game with seven cards in hand.",
            "7:00 AM: b begins the game with seven cards in hand.",
            "7:00 AM: Turn 5: a",
            "7:00 AM: a casts Mystery Beast.",
            "7:00 AM: b is being attacked by Mystery Beast.",
        ])
        lives = {p["name"]: p["life"] for p in states[-1]["players"]}
        self.assertEqual(lives["b"], 20)
        self.assertTrue(any("Mystery Beast" in w for w in sim.warnings))

    def test_transform_renames_in_place(self):
        info = {
            "Delver of Secrets": {
                "type_line": "Creature — Human Wizard // Creature — Human Insect",
                "faces": [
                    {"name": "Delver of Secrets", "type_line": "Creature — Human Wizard",
                     "power": "1", "toughness": "1"},
                    {"name": "Insectile Aberration",
                     "type_line": "Creature — Human Insect",
                     "power": "3", "toughness": "2"},
                ],
            }
        }
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: a casts Delver of Secrets.",
             "7:00 AM: Delver of Secrets transforms into Insectile Aberration."],
            card_info=lambda name: info.get(name),
        )
        battlefield = states[-1]["zones"]["battlefield"]
        self.assertEqual(len(battlefield), 1)
        self.assertEqual(battlefield[0]["name"], "Insectile Aberration")
        self.assertEqual(battlefield[0]["power"], 3)

    def test_transform_takes_power_from_the_back_face_looked_up_by_name(self):
        # The catalog indexes each face as its own card, so the back face is
        # resolvable directly and carries the post-transform power.
        info = {
            "Delver of Secrets": {"type_line": "Creature — Human Wizard",
                                  "power": "1", "toughness": "1"},
            "Insectile Aberration": {"type_line": "Creature — Human Insect",
                                     "power": "3", "toughness": "2"},
        }
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand.",
             "7:00 AM: Turn 3: a",
             "7:00 AM: a casts Delver of Secrets.",
             "7:00 AM: Delver of Secrets transforms into Insectile Aberration.",
             "7:00 AM: b is being attacked by Insectile Aberration.",
             "7:00 AM: Turn 3: b"],
            card_info=lambda name: info.get(name),
        )
        self.assertEqual(sim.warnings, [])
        lives = {p["name"]: p["life"] for p in states[-1]["players"]}
        self.assertEqual(lives["b"], 17)

    def test_countered_permanent_does_not_stay_on_the_battlefield(self):
        info = {"Expedition Map": {"type_line": "Artifact"},
                "Counterspell": {"type_line": "Instant"}}
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand.",
             "7:00 AM: a casts Expedition Map.",
             "7:00 AM: b casts Counterspell targeting Expedition Map.",
             "7:00 AM: b counters Expedition Map with Counterspell."],
            card_info=lambda name: info.get(name),
        )
        names = [o["name"] for o in states[-1]["zones"]["battlefield"]]
        self.assertNotIn("Expedition Map", names)

    def test_opponent_hand_is_face_down(self):
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand."],
            hero="a",
        )
        state = states[-1]
        self.assertEqual(state["localSeat"], 1)
        for seat_cards in state["zones"]["hands"].values():
            for card in seat_cards:
                self.assertTrue(card["faceDown"])
                self.assertNotIn("name", card)

    def test_player_name_ocr_variants_map_to_one_seat(self):
        sim, states = self._sim([
            "7:00 AM: BuzzCaldera begins the game with seven cards in hand.",
            "7:00 AM: Buz2Caldera draws a card.",
            "7:00 AM: Buz2zCaldera draws a card.",
        ])
        self.assertEqual(len(states[-1]["players"]), 1)
        self.assertEqual(states[-1]["players"][0]["handCount"], 9)

    def test_seeded_players_all_appear_in_the_very_first_snapshot(self):
        # The XMage mirror creates its seats once, from the state it opens
        # with, so a player who acts later must already be present.
        sim = GameSimulator(deck_size=60, hero="BuzzCaldera")
        sim.seed_players(["golubtsov", "BuzzCaldera"])
        state = sim.apply(parse_entry("7:00 AM: Turn 1: BuzzCaldera"))[0]
        self.assertEqual([p["name"] for p in state["players"]],
                         ["golubtsov", "BuzzCaldera"])
        self.assertEqual(state["localSeat"], 2)

    def test_a_duel_never_grows_a_third_seat(self):
        sim = GameSimulator(deck_size=60)
        sim.seed_players(["golubtsov", "BuzzCaldera"])
        # A badly mangled name must snap to a known seat, not open a new one.
        state = sim.apply(parse_entry("7:00 AM: 9olub7sov draws a card."))[0]
        self.assertEqual(len(state["players"]), 2)

    def test_a_game_starts_on_turn_one(self):
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand."])
        self.assertEqual(states[-1]["turnNumber"], 1)

    def test_snapshot_matches_the_mirror_state_schema(self):
        sim, states = self._sim([
            "7:00 AM: a begins the game with seven cards in hand.",
            "7:00 AM: Turn 1: a",
        ])
        state = states[-1]
        for field in ("seq", "gameInstance", "matchId", "turnNumber", "phase",
                      "step", "activeSeat", "players", "zones"):
            self.assertIn(field, state)
        for zone in ("battlefield", "hands", "graveyards", "exile", "libraries"):
            self.assertIn(zone, state["zones"])
        # Hands, graveyards, and libraries are keyed by stringified seat.
        self.assertTrue(all(k.isdigit() for k in state["zones"]["hands"]))
        json.dumps(state)  # must be serializable for mirror_states.jsonl


class CrossCaptureCompareTest(unittest.TestCase):
    """Two captures of the same match must decode to the same game."""

    def game(self, bundle="a", life=20, extra_event=False):
        events = [{"type": "TURN", "turn": 1, "player": "a", "text": "Turn 1: a",
                   "videoTime": 1.0},
                  {"type": "PLAY_LAND", "player": "a", "card": "Island",
                   "text": "a plays Island.", "videoTime": 2.0}]
        if extra_event:
            events.append({"type": "DRAW", "player": "a", "count": 1,
                           "text": "a draws a card."})
        states = [{"seq": 1, "turnNumber": 1, "phase": "Phase_Main1",
                   "step": None, "activeSeat": 1, "videoTime": 2.0,
                   "players": [{"seat": 1, "name": "a", "life": life,
                                "libraryCount": 50, "handCount": 6}],
                   "zones": {"battlefield": [{"name": "Island", "tapped": False,
                                              "controllerSeat": 1}],
                             "graveyards": {}},
                   "sourceEvent": {"text": "a plays Island."}}]
        return {"bundle": bundle, "events": events, "states": states}

    def test_identical_games_compare_equal_despite_different_timestamps(self):
        from magic_cabt.mtgo_video.compare import compare_games

        left = self.game("1080p")
        right = self.game("720p")
        # Frame sampling legitimately shifts video timestamps between
        # captures; that must not count as a difference.
        right["events"][0]["videoTime"] = 1.5
        right["states"][0]["videoTime"] = 2.5
        result = compare_games(left, right)
        self.assertTrue(result["identical"], result["differences"])

    def test_a_differing_board_is_reported(self):
        from magic_cabt.mtgo_video.compare import compare_games

        result = compare_games(self.game(), self.game(life=17))
        self.assertFalse(result["identical"])
        self.assertTrue(any(d["kind"] == "states" for d in result["differences"]))

    def test_a_missing_event_is_reported(self):
        from magic_cabt.mtgo_video.compare import compare_games

        result = compare_games(self.game(extra_event=True), self.game())
        self.assertFalse(result["identical"])
        self.assertTrue(any(d["reason"] == "count"
                            for d in result["differences"]))


class CompareStateTest(unittest.TestCase):
    """The log-vs-XMage diff used to verify mirrored behavior."""

    def _state(self, life=20, battlefield=(("Island", False),)):
        return {
            "turnNumber": 2,
            "players": [{"seat": 1, "name": "a", "life": life,
                         "libraryCount": 50, "handCount": 5}],
            "zones": {
                "battlefield": [
                    {"name": n, "tapped": t, "controllerSeat": 1}
                    for n, t in battlefield
                ],
                "hands": {}, "graveyards": {}, "exile": [], "libraries": {"1": 50},
            },
        }

    def _summary(self, life=20, battlefield=(("Island", False),)):
        return {
            "turn": 2,
            "players": [{"name": "a", "life": life, "libraryCount": 50,
                         "handCount": 5, "graveyard": [],
                         "battlefield": [{"name": n, "tapped": t}
                                         for n, t in battlefield]}],
        }

    def test_agreeing_state_has_no_problems(self):
        self.assertEqual(compare_state(self._state(), self._summary()), [])

    def test_life_mismatch_is_reported(self):
        problems = compare_state(self._state(life=20), self._summary(life=17))
        self.assertTrue(any("life" in p for p in problems))

    def test_battlefield_mismatch_is_reported(self):
        problems = compare_state(
            self._state(battlefield=(("Island", False),)),
            self._summary(battlefield=(("Island", True),)),
        )
        self.assertTrue(any("battlefield" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
