"""Fixed-pool, seat-balanced complete-game experiments with retained evidence.

Reuses one warm JVM for throughput, but replaces it after a failed game. A
watchdog kills a blocked JVM. Limits, unknown outcomes and skipped Jev games
never count as draws. Raw snapshots are retained only in local replay files.
"""
from dataclasses import asdict
from itertools import combinations, product
import json
from pathlib import Path
import re
import threading
import time
from magic_cabt.agents import make_agent, is_legal_selection
from magic_cabt.eval.play import acting_seat
from magic_cabt.protocol import CabtBridge, parse_decklist
from .jev import Budget, JevAgent
from .policy import Config, HeuristicAgent
from .state import canonical, digest

ENGINE_REFERENCE = "fd40ad5c29a92cef824cf12ba6d0e4daa25db975"
PLAYER_NAMES = ("P0", "P1")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_hash():
    package = Path(__file__).resolve().parents[1]
    sources = {"python/" + str(p.relative_to(package)): p.read_text(encoding="utf-8")
               for p in sorted(package.rglob("*.py"))}
    root = package.parent.parent
    overlay = root / "Mage.Server.Plugins/Mage.Player.AI/src/main/java"
    sources.update({str(p.relative_to(root)): p.read_text(encoding="utf-8")
                    for p in sorted(overlay.rglob("*.java"))})
    return digest(sources)


def load_pool(path):
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    decks = pool.get("decks")
    if not isinstance(decks, dict) or not 2 <= len(decks) <= 16:
        raise ValueError("A fixed pool must contain 2..16 named decks")
    for name, text in decks.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Deck IDs must be path-safe")
        entries = parse_decklist(text)
        if sum(e["count"] for e in entries) != 60:
            raise ValueError("This experiment requires 60-card decks: " + name)
        counts = {}
        for entry in entries:
            counts[entry["name"]] = counts.get(entry["name"], 0) + entry["count"]
        basics = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
        if any(count > 4 for card, count in counts.items() if card not in basics):
            raise ValueError("Four-copy pool restriction violated: " + name)
    return pool


def schedule(pool, agents, seeds):
    if len(set(agents)) != len(agents) or len(agents) < 2:
        raise ValueError("Provide at least two distinct agents")
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct integer seeds")
    games = []
    for a, b in combinations(agents, 2):
        for deck_a, deck_b in product(sorted(pool["decks"]), repeat=2):
            for seed in seeds:
                for swap in (False, True):
                    players, decks = [a, b], [deck_a, deck_b]
                    if swap:
                        players.reverse()
                        decks.reverse()
                    games.append({"id": "game-%05d" % len(games), "agents": players,
                                  "decks": decks, "seed": seed, "startingSeat": 0,
                                  "block": [a, b, deck_a, deck_b, seed], "swap": swap})
    return games


def classify_result(result, max_turns):
    final = result.get("finalState") or {}
    winner = result.get("winner")
    if isinstance(winner, str) and re.search(r"winner|\bwon\b", winner, re.I):
        matches = [i for i, name in enumerate(PLAYER_NAMES)
                   if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", winner)]
        if len(matches) == 1:
            return "completed", matches[0]
    if final.get("turnNumber", 0) >= max_turns:
        return "turn-limit", None
    if winner == "Game is a draw" and final.get("gameEnded") is True:
        return "completed", None
    return "unresolved-outcome", None


def _starting_selection(observation, starting_seat):
    current, prompt = observation.get("current", {}), observation.get("select", {})
    if current.get("activePlayerId") is not None or current.get("step") is not None:
        return None
    players = current.get("players") or []
    if any(p.get("handCount") for p in players) or prompt.get("type") != "TARGET":
        return None
    identities = {p.get("playerId") for p in players}
    options = prompt.get("option") or []
    if len(options) != 2 or {o.get("payload", {}).get("targetId") for o in options} != identities:
        return None
    target = next(p["playerId"] for p in players if p["name"] == PLAYER_NAMES[starting_seat])
    return [next(i for i, option in enumerate(options) if option["payload"]["targetId"] == target)]


def build_agent(spec, seed, catalog, configs, api_key, budget):
    if spec in ("heuristic", "candidate", "champion", "jev"):
        heuristic = HeuristicAgent(config=configs.get(spec, Config()), catalog=catalog)
        return JevAgent(heuristic, api_key=api_key, budget=budget) if spec == "jev" else heuristic
    return make_agent(spec, seed=seed)


def play_one(bridge, spec, pool, agents, out, max_turns=80, max_decisions=5000,
             wall_seconds=90.0):
    out = Path(out)
    started = time.perf_counter()
    expired = threading.Event()

    def expire():
        expired.set()
        process = getattr(bridge, "_process", None)
        if process is not None and process.poll() is None:
            process.kill()

    timer = threading.Timer(wall_seconds, expire)
    timer.daemon = True
    timer.start()
    report = dict(spec, status="error", winnerSeat=None, decisions=0, elapsedSeconds=0.0,
                  fallbackDecisions=0, error=None, replay=spec["id"] + ".jsonl")
    try:
        with (out / report["replay"]).open("w", encoding="utf-8") as replay:
            replay.write(canonical({"recordType": "experiment-header", "game": spec,
                                    "engineReference": ENGINE_REFERENCE}) + "\n")
            response = bridge.game_start(*(pool["decks"][name] for name in spec["decks"]),
                                         player_names=PLAYER_NAMES, seed=spec["seed"],
                                         max_turns=max_turns, decision_timeout_seconds=30)
            while not bridge.finished:
                if expired.is_set():
                    report["status"] = "timeout"
                    break
                if report["decisions"] >= max_decisions:
                    report["status"] = "decision-limit"
                    break
                observation = response["observation"]
                seat = acting_seat(observation, PLAYER_NAMES)
                start_choice = _starting_selection(observation, spec["startingSeat"])
                selected = start_choice if start_choice is not None else agents[seat].select(observation)
                annotation = {"policy": "fixed-starting-seat", "selection": selected} if start_choice is not None else \
                    getattr(agents[seat], "last_decision", None)
                if not is_legal_selection(selected, observation["select"]):
                    raise ValueError("Agent returned invalid selection: %r" % selected)
                if annotation and annotation.get("fallback"):
                    report["fallbackDecisions"] += 1
                replay.write(canonical({"gameId": spec["id"], "sequence": response.get("sequence"),
                                        "player": seat, "observation": observation, "selected": selected,
                                        "explanation": annotation}) + "\n")
                replay.flush()
                response = bridge.game_select(selected)
                report["decisions"] += 1
            if bridge.finished:
                report["status"], report["winnerSeat"] = classify_result(bridge.result or {}, max_turns)
                replay.write(canonical({"gameId": spec["id"], "result": bridge.result,
                                        "experimentStatus": report["status"]}) + "\n")
                report["turns"] = ((bridge.result or {}).get("finalState") or {}).get("turnNumber")
            elif not expired.is_set():
                bridge.game_finish()
    except Exception as error:
        report["status"] = "timeout" if expired.is_set() else "error"
        report["error"] = type(error).__name__ + ": " + str(error)[:1000]
    finally:
        timer.cancel()
        report["elapsedSeconds"] = round(time.perf_counter() - started, 6)
        report["jev"] = [dict(agent.stats, provenance=agent.provenance)
                         for agent in agents if isinstance(agent, JevAgent)]
    return report


def summarize(games):
    statuses, scores = {}, {}
    for game in games:
        status = game["status"]
        statuses[status] = statuses.get(status, 0) + 1
        if status != "completed":
            continue
        for seat, name in enumerate(game["agents"]):
            entry = scores.setdefault(name, {"wins": 0, "draws": 0, "losses": 0, "games": 0})
            entry["games"] += 1
            outcome = "draws" if game["winnerSeat"] is None else "wins" if seat == game["winnerSeat"] else "losses"
            entry[outcome] += 1
    for entry in scores.values():
        entry["meanScore"] = (entry["wins"] + entry["draws"] * 0.5) / entry["games"]
    return {"statuses": statuses, "scores": scores,
            "decisions": sum(g.get("decisions", 0) for g in games),
            "interpretation": "Descriptive fixed-pool results. Failures excluded, not wins/draws. No expert-strength claim."}


def run_suite(pool, out, agents=("heuristic", "jev", "first", "random"), seeds=(11,),
              configs=None, api_key=None, budget=None, bridge_factory=CabtBridge,
              max_turns=80, max_decisions=5000, wall_seconds=90.0,
              max_games=None, progress=None):
    if type(max_turns) is not int or max_turns < 1 or type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("Positive turn and decision limits required")
    if max_games is not None and (type(max_games) is not int or max_games < 1):
        raise ValueError("max_games must be a positive integer")
    if not 1 <= wall_seconds <= 3600:
        raise ValueError("Game timeout must be 1..3600 seconds")
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    configs = {name: (configs or {}).get(name, Config()) for name in agents
               if name in ("heuristic", "jev", "candidate", "champion")}
    budget = budget or Budget()
    planned = schedule(pool, agents, seeds)
    selected_schedule = planned if max_games is None else planned[:max_games]
    manifest = {"schemaVersion": 1, "engineReference": ENGINE_REFERENCE,
                "codeHash": code_hash(), "poolHash": digest(pool), "pool": pool,
                "configs": {name: asdict(config) for name, config in configs.items()},
                "agents": list(agents), "seeds": list(seeds), "schedule": planned,
                "maxTurns": max_turns, "maxDecisions": max_decisions,
                "wallSeconds": wall_seconds, "completeSchedule": max_games is None,
                "jevConfigured": bool(api_key), "jevBudget": {"maxCalls": budget.max_calls,
                     "reportedCostStopUsd": budget.max_reported_usd},
                "determinism": "Seeded requests and paired seats. UUID ordering may differ; retained replays are authoritative."}
    atomic_json(output / "manifest.json", manifest)
    bridge, games, catalog = None, [], {}
    try:
        for spec in selected_schedule:
            if "jev" in spec["agents"] and not api_key:
                games.append(dict(spec, status="skipped", reason="OPENROUTER_API_KEY unavailable", decisions=0, winnerSeat=None))
                continue
            if "jev" in spec["agents"] and not budget.available:
                games.append(dict(spec, status="skipped", reason="Jev run budget exhausted or billing unknown", decisions=0, winnerSeat=None))
                continue
            if bridge is None:
                bridge = bridge_factory()
                names = sorted({entry["name"] for text in pool["decks"].values() for entry in parse_decklist(text)})
                cards = bridge.repository_card_data(names)
                catalog = {card["name"]: card for card in cards}
                atomic_json(output / "card_catalog.json", cards)
            players = [build_agent(name, spec["seed"] + i, catalog, configs, api_key, budget)
                       for i, name in enumerate(spec["agents"])]
            game = play_one(bridge, spec, pool, players, output, max_turns, max_decisions, wall_seconds)
            game["jevActuallyCalled"] = any(j["calls"] > 0 for j in game.get("jev", []))
            games.append(game)
            atomic_json(output / "games.json", games)
            if progress:
                progress(game)
            if game["status"] in ("error", "timeout"):
                bridge.close()
                bridge = None
    finally:
        if bridge is not None:
            bridge.close()
        summary = dict(summarize(games), manifestHash=digest(manifest), gamesRecorded=len(games),
                       plannedGames=len(planned), budget=asdict(budget))
        summary["finalCodeHash"] = code_hash()
        summary["codeUnchanged"] = summary["finalCodeHash"] == manifest["codeHash"]
        atomic_json(output / "games.json", games)
        atomic_json(output / "summary.json", summary)
    return summary, games
