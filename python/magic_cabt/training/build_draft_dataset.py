"""Build limited-play datasets (draft picks, deck builds, sideboards).

    magic-cabt-build-draft-dataset --log Player.log --out runs/draft-data

Joins ``ARENA_DRAFT_PACK`` events (pack contents) with ``ARENA_DRAFT_PICK``
events (the human's choice) per draft, accumulating the picked pool, and
extracts limited deck submissions and Bo3 sideboard decisions. Sources may be
raw Arena ``Player.log`` files (``--log``) or already-normalized bundle
directories (``--bundle``).

Fail-fast by design: a pick whose cards are not part of the joined pack means
the join is wrong and raises; a pick with no recorded pack (Arena does not
re-notify the very first pack after a reconnect, and pick one of pack one
arrives before the first ``Draft.Notify``) is counted and skipped, never
fabricated.
"""

import argparse
import json
import os
import sys

from magic_cabt.arena_log import ArenaLogNormalizer

__all__ = [
    "collect_limited_records",
    "dedupe_records",
    "build_parser",
    "main",
]

DATASET_FILES = {
    "draftPick": "draft_picks.jsonl",
    "deckBuild": "deck_builds.jsonl",
    "sideboard": "sideboards.jsonl",
}


def collect_limited_records(events, source, stats=None):
    """Turn normalized events from one source into limited-play records.

    Returns ``{"draftPick": [...], "deckBuild": [...], "sideboard": [...]}``.
    """
    stats = stats if stats is not None else {}

    def bump(key):
        stats[key] = stats.get(key, 0) + 1

    packs, picks, deck_submits = {}, {}, []
    prompts, submits = {}, []
    for event in events:
        event_type = event.get("type")
        if event_type == "ARENA_DRAFT_PACK":
            key = (event.get("draftId"), event.get("packNumber"),
                   event.get("pickNumber"))
            packs.setdefault(key, event.get("packCards") or [])
        elif event_type == "ARENA_DRAFT_PICK":
            key = (event.get("draftId"), event.get("packNumber"),
                   event.get("pickNumber"))
            picks[key] = event.get("pickedCardIds") or []
        elif event_type == "ARENA_DECK_SUBMIT":
            deck_submits.append(event)
        elif event_type == "ARENA_SIDEBOARD_PROMPT":
            prompts[(event.get("matchId"), event.get("msgId"))] = event
        elif event_type == "ARENA_SIDEBOARD_SUBMIT":
            submits.append(event)

    pick_records = []
    pools = {}
    for (draft_id, pack_number, pick_number), picked in sorted(
            picks.items(), key=lambda item: (str(item[0][0]),
                                             item[0][1] or 0,
                                             item[0][2] or 0)):
        pool = pools.setdefault(draft_id, [])
        pack = packs.get((draft_id, pack_number, pick_number))
        if pack is None:
            bump("picks_missing_pack")
        else:
            missing = [card for card in picked if card not in pack]
            if missing:
                raise ValueError(
                    "picked cards %s not offered in pack (draft %s pack %s "
                    "pick %s of %s)" % (missing, draft_id, pack_number,
                                        pick_number, source))
            pick_records.append({
                "kind": "draftPick",
                "source": source,
                "draftId": draft_id,
                "packNumber": pack_number,
                "pickNumber": pick_number,
                "pack": list(pack),
                "picked": list(picked),
                "pool": list(pool),
            })
            bump("picks_compiled")
        pool.extend(picked)

    build_records = []
    for event in deck_submits:
        main_deck = event.get("mainDeckArenaIds") or []
        draft_id, pool = _match_pool(main_deck, pools)
        if draft_id is None:
            bump("deck_builds_without_pool")
        build_records.append({
            "kind": "deckBuild",
            "source": source,
            "eventName": event.get("eventName"),
            "draftId": draft_id,
            "pool": pool,
            "mainDeck": list(main_deck),
            "sideboard": list(event.get("sideboardArenaIds") or []),
        })
        bump("deck_builds_compiled")

    sideboard_records = []
    for event in submits:
        prompt = prompts.get((event.get("matchId"), event.get("respId")))
        if prompt is None:
            bump("sideboards_missing_prompt")
            continue
        record = {
            "kind": "sideboard",
            "source": source,
            "matchId": event.get("matchId"),
            "gameNumber": prompt.get("gameNumber"),
            "offeredDeck": list(prompt.get("deckCards") or []),
            "offeredSideboard": list(prompt.get("sideboardCards") or []),
            "chosenDeck": list(event.get("deckCards") or []),
            "chosenSideboard": list(event.get("sideboardCards") or []),
        }
        record["changed"] = sorted(record["offeredDeck"]) != \
            sorted(record["chosenDeck"])
        sideboard_records.append(record)
        bump("sideboards_compiled")

    return {
        "draftPick": pick_records,
        "deckBuild": build_records,
        "sideboard": sideboard_records,
        "pools": pools,
    }


def _match_pool(main_deck, pools):
    """Attach the source's draft pool whose picks best cover the deck.

    Basic lands are granted outside the pool, so coverage is judged on the
    distinct cards that appear in any pool at all.
    """
    distinct = set(main_deck)
    best_id, best_pool, best_overlap = None, None, 0
    for draft_id, pool in pools.items():
        overlap = len(distinct & set(pool))
        if overlap > best_overlap:
            best_id, best_pool, best_overlap = draft_id, list(pool), overlap
    if best_pool is None or best_overlap * 2 < len(distinct & _all_ids(pools)):
        return None, None
    return best_id, best_pool


def _all_ids(pools):
    ids = set()
    for pool in pools.values():
        ids.update(pool)
    return ids


def dedupe_records(records, stats=None):
    """Drop records duplicated across sources (re-saved copies of one log).

    The same draft seen from two seats has different picks and is kept; the
    identity of a record is its full decision content, not the draft id.
    """
    stats = stats if stats is not None else {}
    seen, unique = set(), []
    for record in records:
        key = json.dumps(
            {k: v for k, v in record.items() if k != "source"},
            sort_keys=True)
        if key in seen:
            stats["duplicates_dropped"] = stats.get(
                "duplicates_dropped", 0) + 1
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _events_from_source(path):
    if os.path.isdir(path):
        events_path = os.path.join(path, "normalized_events.jsonl")
        with open(events_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    normalizer = ArenaLogNormalizer().normalize_file(path)
    return normalizer.records["normalized_events"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-build-draft-dataset",
        description="Build draft-pick / deck-build / sideboard datasets from "
                    "Arena logs or normalized bundles.")
    parser.add_argument("--log", action="append", default=[],
                        help="raw Arena Player.log file (repeatable)")
    parser.add_argument("--bundle", action="append", default=[],
                        help="normalized bundle directory (repeatable)")
    parser.add_argument("--out", required=True,
                        help="output directory for dataset JSONL files")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sources = list(args.log) + list(args.bundle)
    if not sources:
        sys.stderr.write("no --log or --bundle sources given\n")
        return 2

    stats = {}
    collected = {kind: [] for kind in DATASET_FILES}
    global_pools = {}
    for path in sources:
        records = collect_limited_records(
            _events_from_source(path), os.path.basename(path), stats)
        for kind in collected:
            collected[kind].extend(records[kind])
        # Two seats of one pod share a draftId but hold different pools;
        # key by source as well so neither clobbers the other.
        for draft_id, pool in records["pools"].items():
            global_pools["%s:%s" % (os.path.basename(path), draft_id)] = pool
    # A draft and its deck submission often land in different log files
    # (Arena rotates Player.log between sessions); retry unmatched deck
    # builds against every source's pools.
    for record in collected["deckBuild"]:
        if record["draftId"] is None:
            key, pool = _match_pool(record["mainDeck"], global_pools)
            if key is not None:
                record["draftId"] = key.split(":", 1)[1]
                record["pool"] = pool
                stats["deck_builds_pool_recovered"] = stats.get(
                    "deck_builds_pool_recovered", 0) + 1
    for kind in collected:
        collected[kind] = dedupe_records(collected[kind], stats)

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for kind, filename in DATASET_FILES.items():
        target = os.path.join(args.out, filename)
        with open(target, "w", encoding="utf-8") as handle:
            for record in collected[kind]:
                handle.write(json.dumps(record, sort_keys=True,
                                        separators=(",", ":")))
                handle.write("\n")
        counts[filename] = len(collected[kind])
    summary = {"sources": sources, "records": counts, "stats": stats}
    with open(os.path.join(args.out, "draft_dataset_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
