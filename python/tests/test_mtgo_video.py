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

    def test_repeating_turn_sequences_do_not_shift_the_alignment(self):
        # A game log repeats itself every turn, which a plain sequence match
        # can lock onto at the wrong occurrence -- duplicating or swallowing
        # a whole pane of lines. The pane only scrolls forward, so alignment
        # must not slip backwards into an earlier identical stretch.
        def turn(n, player):
            return ["7:0%d AM: Turn %d: %s" % (n, n, player),
                    "7:0%d AM: %s draws a card." % (n, player),
                    "7:0%d AM: %s plays Island." % (n, player)]

        log = turn(1, "alice") + turn(2, "bob") + turn(3, "alice")
        rec = LogReconstructor()
        # Slide a 4-line pane over the log, one line at a time.
        for start in range(0, len(log) - 3):
            rec.feed(log[start:start + 4], timestamp=float(start))
        self.assertEqual(rec.entries, log)

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

    def test_a_repeat_run_is_capped_at_what_was_seen_at_once(self):
        from magic_cabt.mtgo_video.reconstruct import Entry, limit_repeats

        text = "7:03 AM: a casts Tolarian Terror."
        norm = normalize(text)
        # Two copies were on screen together in frames 1-3; a third entry
        # only ever appeared alone, so it is a misread that failed to merge.
        run = [Entry(text, norm, 1.0, (text,), (1, 2, 3)),
               Entry(text, norm, 1.0, (text,), (1, 2, 3)),
               Entry(text, norm, 4.0, (text,), (7,))]
        self.assertEqual(len(limit_repeats(run)), 2)

    def test_a_repeat_run_seen_in_full_is_kept_in_full(self):
        from magic_cabt.mtgo_video.reconstruct import Entry, limit_repeats

        text = "7:03 AM: a casts Tolarian Terror."
        norm = normalize(text)
        # All three were on screen at once, so all three are real.
        run = [Entry(text, norm, 1.0, (text,), (1, 2)) for _ in range(3)]
        self.assertEqual(len(limit_repeats(run)), 3)

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

    def test_the_log_forms_a_second_capture_turned_up(self):
        # Every one of these is a real MTGO line that the grammar did not
        # recognise until a second recording contained it. The blocker-first
        # form matters most: without it a block is invisible, and a blocked
        # attacker's damage is still dealt to the defending player.
        cases = [
            ("7:00 AM: Brinebarrow Intruder blocks Tolarian Terror.",
             {"type": "BLOCK", "blocker": "Brinebarrow Intruder",
              "attacker": "Tolarian Terror"}),
            ("7:00 AM: a puts triggered ability from Faerie Seer onto the stack.",
             {"type": "TRIGGER", "card": "Faerie Seer"}),
            ("7:00 AM: Couldn't put triggered ability from Faerie Seer on the stack.",
             {"type": "TRIGGER_FAILED", "card": "Faerie Seer"}),
            ("7:00 AM: a's Triggered ability from Spellstutter Sprite is "
             "removed from the stack because it has no legal targets.",
             {"type": "TRIGGER_FIZZLED", "card": "Spellstutter Sprite"}),
            ("7:00 AM: a activates Ninjutsu ability of Moon-Circuit Hacker.",
             {"type": "ACTIVATE", "card": "Moon-Circuit Hacker"}),
            ("7:00 AM: a draws their next card.", {"type": "DRAW"}),
            ("7:00 AM: a scrys 2 (1 top, 1 bottom).", {"type": "SCRY"}),
            ("7:00 AM: a returns Faerie Seer to its owner's hand with with "
             "Moon-Circuit Hacker's ability.",
             {"type": "RETURN_HAND", "card": "Faerie Seer"}),
            ("7:00 AM: a exiles Relic of Progenitus with Relic of "
             "Progenitus's ability.",
             {"type": "EXILE_CARD", "card": "Relic of Progenitus"}),
            ("7:00 AM: a puts three energy counters on a.",
             {"type": "COUNTERS_ON", "target": "a"}),
            ("7:00 AM: Chosen mode: Counter target red spell.",
             {"type": "CHOICE"}),
            ("7:00 AM: a has left the game.", {"type": "LEAVE"}),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                event = parse_entry(text)
                for key, value in expected.items():
                    self.assertEqual(event[key], value)

    def test_the_blocker_first_form_does_not_swallow_other_lines(self):
        # "<blocker> blocks <attacker>" is a loose shape; it must not claim a
        # line that the player-first form or another rule owns.
        event = parse_entry("7:00 AM: a blocks Tolarian Terror with Faerie Seer.")
        self.assertEqual(event["blocker"], "Faerie Seer")
        self.assertEqual(event["player"], "a")

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


FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)

STEP_NAMES = ("Untap", "Upkeep", "Draw", "Main", "Begin Combat", "Attack",
              "Block", "Damage", "End Combat", "Main", "End", "Cleanup")

LOG_LINES = (
    "7:01 AM: Turn 3: alice",
    "7:01 AM: alice plays Island.",
    "7:02 AM: alice casts Ponder.",
    "7:02 AM: alice draws a card.",
    "7:03 AM: bob casts Lightning Bolt targeting alice.",
    "7:03 AM: alice is now at 17 life.",
    "7:04 AM: bob attacks with Goblin Guide.",
    "7:04 AM: alice discards Counterspell.",
)


def _font(size):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return None


class SyntheticClient:
    """A drawable stand-in for MTGO's duel scene, in any arrangement.

    The detector's whole job is to survive rearrangement, so the fixture has
    to be rearrangeable: which side the seats are on, which side the log is
    on, how big the client is drawn, and whether a streamer's bright overlay
    sits beside it are all parameters here rather than assumptions.
    """

    def __init__(self, width=1920, height=1080, ui_scale=1.0, seats="left",
                 log="right", letterbox=0, overlay=False, log_offset=0):
        self.width = width
        self.height = height
        self.ui_scale = ui_scale * (width / 1920.0)
        self.seats = seats
        self.log = log
        self.letterbox = letterbox
        self.overlay = overlay
        self.log_offset = log_offset

    def _px(self, value):
        return int(round(value * self.ui_scale))

    def render(self):
        from PIL import Image, ImageDraw

        image = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(image)
        top = self.letterbox
        bottom = self.height - self.letterbox - 1
        # A streamed capture rarely gives MTGO the whole frame: the client is
        # squeezed left and the overlay takes the rest.
        client_right = (self.width - self._px(300) if self.overlay
                        else self.width) - 1
        draw.rectangle([0, top, client_right, bottom], fill=70)

        margin = self._px(200)
        pane_width = self._px(300)
        if self.log == "right":
            pane_x = client_right - pane_width - self._px(30) - self.log_offset
        else:
            pane_x = self._px(30)
        seat_x = (self._px(40) if self.seats == "left"
                  else client_right - margin + self._px(40))
        play_left = margin if self.seats == "left" else self._px(20)
        play_right = (client_right - margin if self.seats == "right"
                      else client_right - self._px(20))
        if self.log == "right":
            play_right = min(play_right, pane_x - self._px(10))
        else:
            play_left = max(play_left, pane_x + pane_width + self._px(10))

        inner = bottom - top
        self._draw_log(draw, image, pane_x, top + self._px(60), pane_width)
        # Seats above and below the board's midline, phase bar under the
        # board, hand row below that -- the modern client's vertical order.
        self._draw_phase_bar(draw, play_left, play_right,
                             top + int(inner * 0.72))
        self._draw_seat(draw, seat_x, top + self._px(60), "alice98", 16)
        self._draw_seat(draw, seat_x, top + int(inner * 0.55), "bobmtgo", 20)
        if self.overlay:
            # A streamer's facecam and scoreboard: bright panels that are not
            # the game log, and big numerals that are not life totals.
            cam_x = client_right + self._px(10)
            draw.rectangle([cam_x, top, self.width - 1, top + self._px(400)],
                           fill=235)
            font = _font(self._px(90))
            if font:
                draw.text((cam_x + self._px(40), top + self._px(500)), "0-0",
                          fill=255, font=font)
        return image

    def _draw_log(self, draw, image, x, y, width):
        font = _font(max(9, self._px(13)))
        line_height = self._px(19)
        # A real log pane is tall and full; a short one is not a useful
        # fixture, because a bright sliver is not what the detector faces.
        lines = list(LOG_LINES) * 3
        height = line_height * (len(lines) + 2)
        draw.rectangle([x, y, x + width, y + height], fill=252)
        # Scrollbar: a darker band at the pane's right, with a bright border
        # column outside it, exactly as MTGO draws it.
        bar = self._px(12)
        draw.rectangle([x + width - bar, y, x + width - 2, y + height], fill=160)
        draw.rectangle([x + width - 1, y, x + width, y + height], fill=252)
        if font:
            for index, line in enumerate(lines):
                draw.text((x + self._px(6), y + line_height * (index + 1)),
                          line, fill=20, font=font)

    def _draw_phase_bar(self, draw, left, right, y):
        font = _font(max(9, self._px(14)))
        if not font:
            return
        step = (right - left) / float(len(STEP_NAMES))
        for index, name in enumerate(STEP_NAMES):
            draw.text((left + step * index + self._px(4), y), name,
                      fill=225, font=font)
        draw.text((left - self._px(160), y), "Turn 3: alice", fill=225,
                  font=font)

    def _draw_seat(self, draw, x, y, name, life):
        avatar = self._px(120)
        draw.rectangle([x, y, x + avatar, y + avatar], fill=60)
        life_font = _font(max(12, self._px(44)))
        name_font = _font(max(8, self._px(13)))
        if life_font:
            draw.text((x + avatar - self._px(60), y + avatar - self._px(56)),
                      str(life), fill=255, font=life_font)
        if name_font:
            # MTGO centres the name under the avatar.
            width = draw.textlength(name, font=name_font)
            draw.text((x + (avatar - width) / 2, y + avatar + self._px(8)),
                      name, fill=245, font=name_font)


def _write(image, path):
    image.save(path)
    return path


@unittest.skipIf(_font(12) is None, "no TrueType font available to draw with")
class LayoutTest(unittest.TestCase):
    """Locating the UI without depending on resolution *or* arrangement."""

    def setUp(self):
        import tempfile

        self.work = tempfile.mkdtemp(prefix="layout_test_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.work, ignore_errors=True)

    def frame(self, name="frame", **kwargs):
        return _write(SyntheticClient(**kwargs).render(),
                      os.path.join(self.work, name + ".png"))

    def test_pane_is_found_and_excludes_the_scrollbar(self):
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(self.frame())
        self.assertTrue(layout.detected)
        # The text column stops before the scrollbar band, which starts 12px
        # in from the pane's right edge at this scale.
        self.assertLess(layout.log_pane.x + layout.log_pane.width, 1902)

    def test_the_log_is_found_wherever_it_is_docked(self):
        from magic_cabt.mtgo_video.layout import build_layout

        left = build_layout(self.frame("left", log="left", seats="right"))
        self.assertTrue(left.detected)
        self.assertLess(left.log_pane.x, 400)
        right = build_layout(self.frame("right", log="right", seats="left"))
        self.assertTrue(right.detected)
        self.assertGreater(right.log_pane.x, 1400)

    def test_seats_are_found_on_whichever_side_they_sit(self):
        from magic_cabt.mtgo_video.layout import build_layout

        for seats, log in (("left", "right"), ("right", "left")):
            layout = build_layout(self.frame(seats + log, seats=seats, log=log))
            self.assertTrue(layout.seats_detected, seats)
            if seats == "left":
                self.assertLess(layout.life_top.x, 400)
            else:
                self.assertGreater(layout.life_top.x, 1400)
            # Opponent above, local player below, never the same panel twice.
            self.assertLess(layout.life_top.y, layout.life_bottom.y)

    def test_a_scaled_up_client_is_still_located(self):
        # MTGO's UI scale, and a windowed client, both change how big the
        # panels are without moving them proportionally.
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(self.frame("big", ui_scale=1.4))
        self.assertTrue(layout.detected)
        self.assertTrue(layout.seats_detected)

    def test_a_streamers_overlay_is_not_mistaken_for_the_ui(self):
        # A facecam is a bright panel and a scoreboard is a pair of big
        # numerals; neither is the game log or a seat.
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(self.frame("stream", overlay=True))
        self.assertTrue(layout.detected)
        # The log is inside the client, left of where the overlay begins.
        self.assertLess(layout.log_pane.x + layout.log_pane.width, 1620)
        self.assertTrue(layout.seats_detected)
        self.assertLess(layout.life_top.x, 400)

    def test_detection_scales_with_capture_resolution(self):
        from magic_cabt.mtgo_video.layout import build_layout

        panes = {}
        for width, height in ((1920, 1080), (2560, 1440)):
            layout = build_layout(self.frame("r%d" % width, width=width,
                                             height=height))
            self.assertTrue(layout.detected, "%dx%d" % (width, height))
            panes[width] = layout.log_pane
        self.assertAlmostEqual(panes[2560].width / panes[1920].width,
                               2560 / 1920, delta=0.06)

    def test_letterboxing_is_trimmed_before_locating_anything(self):
        from magic_cabt.mtgo_video.layout import build_layout

        layout = build_layout(self.frame("letterbox", height=1200,
                                         letterbox=60))
        self.assertEqual(layout.content.y, 60)
        self.assertEqual(layout.content.height, 1080)
        self.assertTrue(layout.detected)

    def test_a_menu_screen_is_not_a_duel(self):
        # A deck editor has a game log but no phase bar, and its pane sits
        # somewhere else entirely; treating it as a duel is what pulled the
        # layout consensus off the board.
        from PIL import Image, ImageDraw

        from magic_cabt.mtgo_video.layout import looks_like_duel

        image = Image.new("L", (1920, 1080), 70)
        draw = ImageDraw.Draw(image)
        draw.rectangle([1500, 60, 1880, 900], fill=252)
        font = _font(14)
        for index, line in enumerate(LOG_LINES):
            draw.text((1510, 80 + 20 * index), line, fill=20, font=font)
        path = _write(image, os.path.join(self.work, "menu.png"))
        self.assertFalse(looks_like_duel(path))
        self.assertTrue(looks_like_duel(self.frame("duel")))

    def test_falls_back_to_proportional_regions_when_nothing_is_found(self):
        from PIL import Image

        from magic_cabt.mtgo_video.layout import build_layout

        path = _write(Image.new("L", (1920, 1080), 80),
                      os.path.join(self.work, "blank.png"))
        layout = build_layout(path)
        self.assertFalse(layout.detected)
        self.assertFalse(layout.seats_detected)
        self.assertGreater(layout.log_pane.width, 0)
        self.assertTrue(layout.notes)

    def test_consensus_spans_the_widest_text_seen_but_not_the_scrollbar(self):
        # Each frame's bounds come from the words in it, so a frame of short
        # lines understates the column: the consensus has to be the union, or
        # the crop clips the last character off every full-width line. It
        # still must not reach the scrollbar, which no text ever sits in.
        from magic_cabt.mtgo_video.layout import (build_layout,
                                                  detect_layout_from_frames)

        paths = [self.frame("c%d" % index, log_offset=index * 2)
                 for index in range(3)]
        layout = detect_layout_from_frames(paths)
        self.assertEqual(layout.samples, 3)
        self.assertTrue(layout.seats_detected)
        widest = max(build_layout(path).log_pane.x + build_layout(path).log_pane.width
                     for path in paths)
        self.assertGreaterEqual(layout.log_pane.x + layout.log_pane.width, widest)
        self.assertLess(layout.log_pane.x + layout.log_pane.width, 1902)


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

    def test_an_attacker_bounced_out_of_combat_deals_no_damage(self):
        # Ninjutsu returns the unblocked attacker to hand and puts the ninja
        # in attacking. Keeping the returned creature's damage *and* missing
        # the ninja's leaves every life total afterwards wrong by the
        # difference, for the rest of the game.
        info = {"Faerie Seer": {"type_line": "Creature — Faerie Wizard",
                                "power": "1", "toughness": "1"},
                "Moon-Circuit Hacker": {"type_line": "Creature — Human Ninja",
                                        "power": "2", "toughness": "1"}}
        sim, states = self._sim(
            ["7:00 AM: a begins the game with seven cards in hand.",
             "7:00 AM: b begins the game with seven cards in hand.",
             "7:00 AM: Turn 5: a",
             "7:00 AM: a casts Faerie Seer.",
             "7:00 AM: b is being attacked by Faerie Seer.",
             "7:00 AM: a returns Faerie Seer to its owner's hand with with "
             "Moon-Circuit Hacker's ability.",
             "7:00 AM: a activates Ninjutsu ability of Moon-Circuit Hacker.",
             "7:00 AM: Turn 5: b"],
            card_info=lambda name: info.get(name),
        )
        lives = {p["name"]: p["life"] for p in states[-1]["players"]}
        self.assertEqual(lives["b"], 18)  # the ninja's 2, not the faerie's 1
        battlefield = [o["name"] for o in states[-1]["zones"]["battlefield"]]
        self.assertEqual(battlefield, ["Moon-Circuit Hacker"])

    def test_a_card_named_after_a_player_never_reaches_the_board(self):
        # MTGO really does print a player's name where a card's belongs.
        # Whatever was cast, it was not a card called "b".
        sim, states = self._sim([
            "7:00 AM: a begins the game with seven cards in hand.",
            "7:00 AM: b begins the game with seven cards in hand.",
            "7:00 AM: Turn 3: a",
            "7:00 AM: a casts b targeting Tolarian Terror.",
        ])
        zones = states[-1]["zones"]
        self.assertEqual(zones["battlefield"], [])
        self.assertEqual(zones["graveyards"], {})
        self.assertTrue(any("unnamed card" in w for w in sim.warnings))

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


class CombinedCheckTest(unittest.TestCase):
    """Running every verification the bundle and the machine allow."""

    def run_check(self, **kwargs):
        import argparse
        import contextlib
        import io
        import tempfile

        from magic_cabt.mtgo_video.__main__ import run_check

        work = tempfile.mkdtemp(prefix="check_test_")
        try:
            args = argparse.Namespace(
                bundle=work, video=None, against=[], classpath=None,
                java="java", cwd=None, sample=1, art_cache=None,
                out=os.path.join(work, "out.json"))
            for key, value in kwargs.items():
                setattr(args, key, value)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    run_check(args)
                except SystemExit:
                    pass
            with open(args.out) as handle:
                return json.load(handle)
        finally:
            import shutil
            shutil.rmtree(work, ignore_errors=True)

    def test_a_bundle_nothing_could_be_run_against_is_not_ok(self):
        # The failure this exists to prevent: five checks, none of them able
        # to run, and a green result at the end of it.
        report = self.run_check()
        self.assertEqual(report["checksRun"], 0)
        self.assertFalse(report["ok"])
        self.assertTrue(all(c["status"] == "skipped" for c in report["checks"]))

    def test_every_skip_says_what_was_missing(self):
        report = self.run_check()
        for check in report["checks"]:
            self.assertTrue(check["detail"],
                            "%s skipped without a reason" % check["check"])

    def test_a_check_that_raises_is_reported_not_swallowed(self):
        # A check that blew up is not a check that passed, and the bundle
        # here has no states for verify to read.
        report = self.run_check(classpath="/nonexistent")
        statuses = {c["check"]: c["status"] for c in report["checks"]}
        self.assertEqual(statuses["verify"], "error")
        self.assertFalse(report["ok"])


class AlignmentTest(unittest.TestCase):
    """Naming the event a violation implies is missing."""

    def test_a_permanent_that_never_arrived_is_named(self):
        from magic_cabt.mtgo_video.rules import propose_repair

        repair = propose_repair(
            {"rule": "missing-permanent", "detail": "..."},
            {"type": "DESTROYED", "card": "Faerie Seer"})
        self.assertEqual(repair["type"], "ENTERS_BATTLEFIELD")
        self.assertEqual(repair["card"], "Faerie Seer")

    def test_an_attacker_that_never_arrived_is_named(self):
        from magic_cabt.mtgo_video.rules import propose_repair

        repair = propose_repair(
            {"rule": "empty-battlefield", "detail": "..."},
            {"type": "ATTACKED_BY", "player": "b",
             "cards": ["Tolarian Terror"]})
        self.assertEqual(repair["card"], "Tolarian Terror")

    def test_damage_in_place_proposes_no_insertion(self):
        # "This card is named after a player" is a line that is wrong where
        # it stands, not a line orphaned by a lost one. Proposing an
        # insertion for it would invent history.
        from magic_cabt.mtgo_video.rules import propose_repair

        self.assertIsNone(propose_repair(
            {"rule": "card-is-player", "detail": "..."},
            {"type": "CAST", "card": "alice"}))

    def test_the_lost_line_is_found_among_the_unparsed_ones(self):
        # A dropped line usually did not vanish -- OCR mangled it into
        # something the grammar could not parse, and it is still there.
        from magic_cabt.mtgo_video.rules import find_lost_line

        events = [
            {"type": "UNPARSED", "videoTime": 10.0,
             "text": "alice plays Volatile Fjord. CIS DAR: Treen 9 AAarchCl"},
            {"type": "UNPARSED", "videoTime": 20.0,
             "text": "something else entirely"},
            {"type": "CAST", "videoTime": 30.0, "text": "alice casts Ponder."},
        ]
        found = find_lost_line({"card": "Volatile Fjord"}, events, before=40.0)
        self.assertIsNotNone(found)
        self.assertIn("Volatile Fjord", found["text"])

    def test_a_line_after_the_violation_is_not_the_cause_of_it(self):
        from magic_cabt.mtgo_video.rules import find_lost_line

        events = [{"type": "UNPARSED", "videoTime": 90.0,
                   "text": "alice plays Volatile Fjord. junk"}]
        self.assertIsNone(
            find_lost_line({"card": "Volatile Fjord"}, events, before=40.0))


class BoardReaderTest(unittest.TestCase):
    """Reading permanents off the screen rather than deriving them."""

    PANEL_TOP = 70
    PANEL_BOTTOM = 150
    CARD_W = 100
    CARD_H = 160

    def setUp(self):
        import tempfile

        self.work = tempfile.mkdtemp(prefix="board_test_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.work, ignore_errors=True)

    def art_of(self, seed):
        """A distinctive little picture, standing in for a card's art."""
        from PIL import Image

        image = Image.new("RGB", (60, 40))
        pixels = image.load()
        for y in range(40):
            for x in range(60):
                pixels[x, y] = ((seed * 37 + x * 3) % 256,
                                (seed * 91 + y * 5) % 256,
                                (seed * 53 + x * y) % 256)
        return image

    def frame(self, top_cards=(), bottom_cards=(), tapped=False):
        """A board: two panels, cards drawn as frame + art + text box."""
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (1200, 700), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 20, 1100, 330], fill=(self.PANEL_TOP,) * 3)
        draw.rectangle([100, 336, 1100, 640], fill=(self.PANEL_BOTTOM,) * 3)
        for panel_y, cards in ((30, top_cards), (350, bottom_cards)):
            for index, seed in enumerate(cards):
                self._draw_card(image, draw, 300 + index * (self.CARD_W + 12),
                                panel_y, seed, tapped)
        return image

    def _draw_card(self, image, draw, x, y, seed, tapped):
        width, height = (self.CARD_H, self.CARD_W) if tapped else (self.CARD_W,
                                                                   self.CARD_H)
        draw.rectangle([x, y, x + width, y + height], fill=(240, 240, 240))
        art = self.art_of(seed)
        if tapped:
            art = art.rotate(-90, expand=True)
            box = (x + int(width * 0.13), y + int(height * 0.05),
                   x + int(width * 0.55), y + int(height * 0.95))
        else:
            box = (x + int(width * 0.05), y + int(height * 0.13),
                   x + int(width * 0.95), y + int(height * 0.55))
        image.paste(art.resize((box[2] - box[0], box[3] - box[1])), box[:2])

    def index_of(self, names):
        from magic_cabt.mtgo_video.art import ArtIndex, signature

        index = ArtIndex(os.path.join(self.work, "art.json"))
        for seed, name in names.items():
            index.signatures[name] = signature(self.art_of(seed))
        return index

    def layout(self):
        from magic_cabt.mtgo_video.layout import Layout
        from magic_cabt.mtgo_video.regions import Region

        return Layout(
            frame_width=1200, frame_height=700,
            content=Region(x=0, y=0, width=1200, height=700),
            log_pane=Region(x=1150, y=0, width=40, height=600),
            life_top=Region(x=10, y=40, width=40, height=30),
            life_bottom=Region(x=10, y=500, width=40, height=30),
            name_top=Region(x=10, y=75, width=60, height=14),
            name_bottom=Region(x=10, y=535, width=60, height=14),
            scale=1.0, detected=False, seats_detected=True,
            phase_bar=Region(x=120, y=660, width=900, height=14))

    def observe(self, image, index, unit=None):
        from magic_cabt.mtgo_video import board

        path = os.path.join(self.work, "frame.png")
        image.save(path)
        return board.observe(path, self.layout(), index, unit=unit)

    def test_each_panels_cards_are_found_and_named(self):
        index = self.index_of({1: "Alpha", 2: "Beta", 3: "Gamma"})
        seen = self.observe(self.frame(top_cards=(1,), bottom_cards=(2, 3)),
                            index)
        self.assertIsNotNone(seen)
        top = [card.name for card in seen["zones"]["top"]["cards"]]
        bottom = [card.name for card in seen["zones"]["bottom"]["cards"]]
        self.assertEqual(top, ["Alpha"])
        self.assertEqual(sorted(bottom), ["Beta", "Gamma"])

    def test_a_card_outside_the_vocabulary_is_not_guessed_at(self):
        # The vocabulary is what the log named. A permanent that matches
        # none of it is exactly what a dropped "casts X" looks like, and
        # naming it after its nearest neighbour would erase that.
        index = self.index_of({1: "Alpha", 2: "Beta"})
        seen = self.observe(self.frame(top_cards=(9,)), index)
        names = [card.name for card in seen["zones"]["top"]["cards"]]
        self.assertEqual(names, [None])

    def test_a_tapped_card_is_read_through_its_rotation(self):
        # MTGO turns a tapped permanent on its side. Read as though it were
        # upright, its art window lands on rules text and bare panel.
        index = self.index_of({1: "Alpha", 2: "Beta", 3: "Gamma"})
        seen = self.observe(self.frame(bottom_cards=(2,), tapped=True), index,
                            unit=self.CARD_W)
        names = [card.name for card in seen["zones"]["bottom"]["cards"]]
        self.assertEqual(names, ["Beta"])

    def test_an_empty_panel_reports_nothing_rather_than_the_flattest_card(self):
        index = self.index_of({1: "Alpha", 2: "Beta"})
        seen = self.observe(self.frame(bottom_cards=(1,)), index)
        self.assertEqual(seen["zones"]["top"]["cards"], [])

    def test_only_permanents_seen_more_than_once_are_reported(self):
        from magic_cabt.mtgo_video.board import _confirm

        findings = [
            {"seat": 1, "card": "Alpha", "x": 300, "distance": 12.0,
             "stateIndex": 1, "videoTime": 10.0},
            {"seat": 1, "card": "Alpha", "x": 306, "distance": 11.0,
             "stateIndex": 2, "videoTime": 20.0},
            {"seat": 1, "card": "Beta", "x": 700, "distance": 25.0,
             "stateIndex": 1, "videoTime": 10.0},
        ]
        confirmed, fleeting = _confirm(findings)
        self.assertEqual([f["card"] for f in confirmed], ["Alpha"])
        self.assertEqual(confirmed[0]["sightings"], 2)
        self.assertEqual([f["card"] for f in fleeting], ["Beta"])

    def test_the_same_card_seen_in_two_places_is_two_findings(self):
        from magic_cabt.mtgo_video.board import _confirm

        findings = [
            {"seat": 1, "card": "Alpha", "x": 300, "distance": 12.0,
             "stateIndex": 1, "videoTime": 10.0},
            {"seat": 1, "card": "Alpha", "x": 900, "distance": 12.0,
             "stateIndex": 2, "videoTime": 20.0},
        ]
        confirmed, fleeting = _confirm(findings)
        self.assertEqual(confirmed, [])
        self.assertEqual(len(fleeting), 2)


class HudCorrectionTest(unittest.TestCase):
    """Adopting the life the footage shows when the derivation cannot be."""

    def states(self, derived=(17, 20), later=(17, 20)):
        def snapshot(life, kind):
            return {"sourceEvent": {"type": kind},
                    "players": [{"seat": 1, "name": "a", "life": life[0]},
                                {"seat": 2, "name": "b", "life": life[1]}]}
        return [snapshot(derived, "COMBAT_DAMAGE"),
                snapshot(later, "CAST"),
                snapshot(later, "CAST")]

    def correct(self, states, shown, index=0):
        from magic_cabt.mtgo_video.hud import _correct_from_hud

        return _correct_from_hud(states, index, {1: "top", 2: "bottom"},
                                 (300.0, shown))

    def test_an_overstated_attack_is_corrected_and_carried_forward(self):
        # MTGO logs nothing when an attacker is killed mid-combat, so the
        # derived damage can be too high. The screen says what really
        # happened, and every later state inherits the difference.
        states = self.states(derived=(14, 20), later=(14, 20))
        result = self.correct(states, {"top": 16, "bottom": 20})
        self.assertIsNotNone(result)
        self.assertEqual(result["correctedBy"], {"1": 2})
        self.assertEqual([s["players"][0]["life"] for s in states], [16, 16, 16])
        self.assertEqual(states[0]["lifeSource"], "hud")

    def test_a_reading_showing_more_damage_is_not_adopted(self):
        # Less damage than derived is the failure MTGO's silence produces.
        # More is some other story, and guessing at it is not an improvement.
        states = self.states(derived=(14, 20))
        self.assertIsNone(self.correct(states, {"top": 11, "bottom": 20}))
        self.assertEqual(states[0]["players"][0]["life"], 14)

    def test_an_unreadable_seat_is_not_a_correction(self):
        states = self.states(derived=(14, 20))
        self.assertIsNone(self.correct(states, {"top": None, "bottom": 20}))

    def test_agreement_that_came_from_the_hud_is_counted_separately(self):
        # A state whose life was taken from the screen agrees with the screen
        # by construction; counting it as independent evidence would let the
        # check confirm its own input.
        from magic_cabt.mtgo_video import hud

        # Screen says 16/20 throughout. The first state's life was taken from
        # it, the second derived and agreeing, the third derived and wrong.
        states = self.states(derived=(16, 20), later=(16, 20))
        states[0]["lifeSource"] = "hud"
        states[2]["players"][0]["life"] = 15
        for index, state in enumerate(states):
            state["videoTime"] = float(index)
        calls = {"life": lambda *a, **k: {"top": 16, "bottom": 20}}
        original_life, original_slots = hud.read_life, hud.map_slots_to_seats
        hud.read_life = lambda *a, **k: calls["life"]()
        hud.map_slots_to_seats = lambda *a, **k: {"top": 1, "bottom": 2}
        try:
            report = hud.crosscheck_life("video.mp4", states)
        finally:
            hud.read_life, hud.map_slots_to_seats = original_life, original_slots
        self.assertEqual(report["lifeReadingsTakenFromHud"], 2)
        # Excluding the two readings that came from the screen makes the
        # remaining agreement rate lower, not higher.
        self.assertLess(report["independentAgreementRate"],
                        report["agreementRate"])


class RulesCheckTest(unittest.TestCase):
    """Each event against the board XMage holds the moment before it."""

    def board(self, battlefield=(("alice", "Grizzly Bears"),), hand=5,
              library=40, graveyard=(), turn=3):
        from magic_cabt.mtgo_video.rules import _Board

        players = {}
        for seat in ("alice", "bob"):
            players[seat] = {
                "name": seat, "life": 20, "handCount": hand,
                "libraryCount": library, "battlefield": [],
                "graveyard": [name for owner, name in graveyard
                              if owner == seat],
            }
        for owner, name in battlefield:
            players[owner]["battlefield"].append({"name": name, "tapped": False})
        return _Board({"turn": turn, "players": list(players.values())})

    def check(self, event, board=None):
        from magic_cabt.mtgo_video.rules import check_event

        return check_event(event, board if board is not None else self.board())

    def test_destroying_a_creature_that_is_there_is_fine(self):
        self.assertEqual(
            self.check({"type": "DESTROYED", "card": "Grizzly Bears"}), [])

    def test_destroying_a_creature_that_is_not_there_is_reported(self):
        # The failure this exists to catch: a missed "casts X" line leaves a
        # later "X is destroyed" with nothing to destroy, and neither the
        # XMage render check nor the life crosscheck can see it.
        problems = self.check({"type": "DESTROYED", "card": "Lightning Bolt"})
        self.assertEqual([p["rule"] for p in problems], ["missing-permanent"])

    def test_sacrificing_the_other_seats_permanent_is_reported(self):
        problems = self.check({"type": "SACRIFICE", "player": "bob",
                               "card": "Grizzly Bears"})
        self.assertEqual([p["rule"] for p in problems], ["wrong-controller"])

    def test_discarding_from_an_empty_hand_is_reported(self):
        problems = self.check({"type": "DISCARD", "player": "alice",
                               "card": "Ponder"}, self.board(hand=0))
        self.assertEqual([p["rule"] for p in problems], ["empty-hand"])

    def test_drawing_more_than_the_library_holds_is_reported(self):
        problems = self.check({"type": "DRAW", "player": "alice", "count": 2},
                              self.board(library=1))
        self.assertEqual([p["rule"] for p in problems], ["empty-library"])

    def test_an_ability_may_be_activated_from_a_graveyard(self):
        # Flashback and unearth activate from the graveyard, so a source that
        # is not on the battlefield is not by itself wrong.
        board = self.board(graveyard=(("alice", "Deep Analysis"),))
        self.assertEqual(self.check({"type": "ACTIVATE", "player": "alice",
                                     "card": "Deep Analysis"}, board), [])

    def test_an_ability_of_a_card_nobody_has_is_reported(self):
        problems = self.check({"type": "ACTIVATE", "player": "alice",
                               "card": "Mox Sapphire",
                               "text": "alice activates an ability of Mox Sapphire."})
        self.assertEqual([p["rule"] for p in problems], ["missing-source"])

    def test_an_ability_activated_from_hand_is_not_a_missing_source(self):
        # Ninjutsu is activated from the hand, whose contents MTGO never
        # logs, so the source being nowhere on the board says nothing.
        self.assertEqual(self.check({
            "type": "ACTIVATE", "player": "alice",
            "card": "Moon-Circuit Hacker",
            "text": "alice activates Ninjutsu ability of Moon-Circuit Hacker."
        }), [])

    def test_a_turn_that_skips_ahead_is_reported(self):
        problems = self.check({"type": "TURN", "turn": 7, "player": "alice"})
        self.assertEqual([p["rule"] for p in problems], ["turn-skipped"])

    def test_a_turn_that_goes_backwards_is_reported(self):
        problems = self.check({"type": "TURN", "turn": 2, "player": "alice"})
        self.assertEqual([p["rule"] for p in problems], ["turn-went-backwards"])

    def test_a_misread_player_name_is_reported_as_a_misread(self):
        # "Buz2Caldera" for "BuzzCaldera": near enough to name the seat it
        # came from, which is what makes it worth reporting rather than
        # quietly accepting a third player into a two-player game.
        problems = self.check({"type": "DRAW", "player": "a1ice", "count": 1})
        self.assertEqual([p["rule"] for p in problems], ["garbled-player"])
        self.assertIn("alice", problems[0]["detail"])

    def test_a_target_that_is_a_player_is_fine(self):
        self.assertEqual(self.check({"type": "CAST", "player": "alice",
                                     "card": "Lightning Bolt",
                                     "targets": ["bob"]}), [])

    def test_a_target_with_ocr_junk_stuck_to_it_is_reported(self):
        problems = self.check({"type": "CAST", "player": "alice",
                               "card": "Thought Scour",
                               "targets": ["bob. 2"]})
        self.assertEqual([p["rule"] for p in problems], ["garbled-target"])

    def test_attacking_with_a_creature_that_is_not_in_play_is_reported(self):
        problems = self.check({"type": "ATTACKED_BY", "player": "bob",
                               "cards": ["Grizzly Bears", "Air Elemental"]})
        self.assertEqual([p["rule"] for p in problems], ["missing-permanent"])
        self.assertIn("Air Elemental", problems[0]["detail"])


if __name__ == "__main__":
    unittest.main()
