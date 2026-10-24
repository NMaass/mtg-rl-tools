"""One predeclared refinement cycle: propose, regress, tune, hold out, gate.

Only bounded module weights are editable. No generated source is executed, no
regression suite is rewritten, and no candidate is promoted on training games.
A conservative one-sided Hoeffding bound is applied to seed-block scores; all
matchups sharing a seed belong to the same block. Repeated runs need fresh
holdout seeds; this module does not license adaptive reuse of a test set.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
from .experiments import atomic_json, code_hash, run_suite, schedule
from .policy import Config, HeuristicAgent, MODULES
from .state import digest


def propose(champion):
    proposals = []
    for module in ("tactics", "mana", "role", "threats", "plans"):
        for multiplier in (1.25, 0.75):
            weights = dict(champion.weights)
            weights[module] = min(4.0, max(0.0, weights[module] * multiplier))
            if weights == champion.weights:
                continue
            candidate = Config(weights, champion.uncertainty_margin, champion.version + "-" + module + "-" + str(multiplier))
            proposals.append({"config": candidate, "hypothesis": "Test %s emphasis at %.2fx without changing other modules" % (module, multiplier)})
    return proposals


def regressions(config, suite):
    cases = suite.get("cases") or []
    if not cases:
        raise ValueError("An empty regression suite cannot certify a candidate")
    failed = []
    for case in cases:
        policy = HeuristicAgent(config, suite.get("catalog") or {})
        selected = policy.select(case["observation"])
        acceptable = case["acceptableSelections"]
        if selected not in acceptable:
            failed.append({"id": case["id"], "actual": selected, "acceptable": acceptable})
    return {"evaluated": len(cases), "passed": len(cases) - len(failed), "failed": failed,
            "suiteHash": digest(suite), "configHash": config.fingerprint,
            "suiteKind": suite.get("kind", "unspecified")}


def promotion_gate(candidate, champion, suite, pool, tuning_seeds, manifest, games,
                   required_seed_blocks=30, family_size=1, alpha=0.05,
                   expected_code_hash=None):
    reasons = []
    regression = regressions(candidate, suite)
    if regression["failed"]:
        reasons.append("tactical-regressions")
    holdout_seeds = manifest.get("seeds") or []
    if not holdout_seeds or set(tuning_seeds) & set(holdout_seeds):
        reasons.append("holdout-seeds-not-disjoint")
    if manifest.get("poolHash") != digest(pool):
        reasons.append("pool-mismatch")
    if expected_code_hash is not None and manifest.get("codeHash") != expected_code_hash:
        reasons.append("code-mismatch")
    expected_configs = {"candidate": asdict(candidate), "champion": asdict(champion)}
    if manifest.get("configs") != expected_configs:
        reasons.append("policy-config-mismatch")
    if manifest.get("agents") != ["candidate", "champion"]:
        reasons.append("wrong-comparison")
    if manifest.get("completeSchedule") is not True:
        reasons.append("incomplete-schedule")
    expected = schedule(pool, ("candidate", "champion"), holdout_seeds) if holdout_seeds else []
    if manifest.get("schedule") != expected:
        reasons.append("schedule-mismatch")
    if len(games) != len(expected) or any(g.get("status") != "completed" for g in games):
        reasons.append("incomplete-or-failed-games")
    if any(g.get("winnerSeat") is not None and (type(g.get("winnerSeat")) is not int or g["winnerSeat"] not in (0, 1)) for g in games):
        reasons.append("invalid-winner-seat")
    if len({g.get("id") for g in games}) != len(games):
        reasons.append("duplicate-games")
    for planned, actual in zip(expected, games):
        if any(actual.get(k) != v for k, v in planned.items()):
            reasons.append("game-identity-mismatch")
            break
    if type(family_size) is not int or family_size < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid multiplicity/alpha")
    blocks = {}
    for game in games:
        if game.get("status") != "completed" or "candidate" not in game.get("agents", []):
            continue
        seat = game["agents"].index("candidate")
        score = 0.5 if game.get("winnerSeat") is None else float(game["winnerSeat"] == seat)
        blocks.setdefault(game["seed"], []).append(score)
    block_scores = [sum(values) / len(values) for values in blocks.values()]
    mean = sum(block_scores) / len(block_scores) if block_scores else None
    lower = mean - math.sqrt(math.log(family_size / alpha) / (2 * len(block_scores))) if block_scores else None
    if len(block_scores) < required_seed_blocks:
        reasons.append("insufficient-independent-seed-blocks")
    if lower is None or lower <= 0.5:
        reasons.append("improvement-not-verified")
    return {"promote": not reasons, "reasons": list(dict.fromkeys(reasons)),
            "regressions": regression, "seedBlocks": len(block_scores), "meanScore": mean,
            "oneSidedLowerBound": lower, "familySize": family_size, "alpha": alpha,
            "assumption": "Independent predeclared seed blocks; scores include all planned matchups and both seats"}


def run_refinement(pool, suite_path, out, candidate_count=2, tuning_seeds=(101,),
                   holdout_seeds=(201, 202), runner=run_suite):
    if type(candidate_count) is not int or not 1 <= candidate_count <= 10:
        raise ValueError("candidate_count must be 1..10")
    if not tuning_seeds or not holdout_seeds or set(tuning_seeds) & set(holdout_seeds):
        raise ValueError("Use nonempty, disjoint tuning and holdout seeds")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new refinement directory")
    out.mkdir(parents=True, exist_ok=True)
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    suite_hash, frozen_code = digest(suite), code_hash()
    champion = Config()
    baseline_regressions = regressions(champion, suite)
    if baseline_regressions["failed"]:
        raise ValueError("Champion fails the frozen regression suite")
    proposals = propose(champion)[:candidate_count]
    atomic_json(out / "plan.json", {"suiteHash": suite_hash, "poolHash": digest(pool),
                "codeHash": frozen_code, "tuningSeeds": list(tuning_seeds),
                "holdoutSeeds": list(holdout_seeds), "familySize": candidate_count,
                "proposals": [{"config": asdict(p["config"]), "hypothesis": p["hypothesis"]} for p in proposals]})
    atomic_json(out / "champion.json", asdict(champion))
    # Run the implementation regressions before generating or accepting results.
    tests = Path(__file__).resolve().parents[2] / "tests"
    with (out / "regression-tests.log").open("w", encoding="utf-8") as log:
        check = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(tests),
                                "-p", "test_heuristic*.py", "-v"], stdout=log, stderr=subprocess.STDOUT,
                               timeout=120, check=False)
    if check.returncode:
        result = {"promoted": False, "reason": "implementation-regression-tests-failed"}
        atomic_json(out / "result.json", result)
        return result
    evaluated = []
    for index, proposal in enumerate(proposals):
        config = proposal["config"]
        regression = regressions(config, suite)
        entry = {"configHash": config.fingerprint, "hypothesis": proposal["hypothesis"],
                 "regressions": regression, "tuningScore": None}
        if not regression["failed"]:
            summary, games = runner(pool, out / ("tuning-%02d" % index),
                                    agents=("candidate", "champion"), seeds=tuple(tuning_seeds),
                                    configs={"candidate": config, "champion": champion})
            if all(g["status"] == "completed" for g in games) and games:
                entry["tuningScore"] = summary["scores"]["candidate"]["meanScore"]
        evaluated.append(entry)
    eligible = [i for i, e in enumerate(evaluated) if e["tuningScore"] is not None]
    if not eligible:
        result = {"promoted": False, "reason": "no-candidate-passed-tuning", "candidates": evaluated}
    else:
        best = max(eligible, key=lambda i: (evaluated[i]["tuningScore"], -i))
        candidate = proposals[best]["config"]
        _, games = runner(pool, out / "holdout", agents=("candidate", "champion"),
                          seeds=tuple(holdout_seeds), configs={"candidate": candidate, "champion": champion})
        manifest = json.loads((out / "holdout" / "manifest.json").read_text())
        gate = promotion_gate(candidate, champion, suite, pool, tuning_seeds, manifest, games,
                              family_size=candidate_count, expected_code_hash=frozen_code)
        if code_hash() != frozen_code or digest(json.loads(Path(suite_path).read_text())) != suite_hash:
            gate["promote"] = False
            gate["reasons"].append("code-or-suite-changed-during-run")
        result = {"promoted": gate["promote"], "selectedCandidate": best, "candidates": evaluated,
                  "gate": gate, "championBefore": champion.fingerprint,
                  "championAfter": candidate.fingerprint if gate["promote"] else champion.fingerprint}
        if gate["promote"]:
            atomic_json(out / "champion.json", asdict(candidate))
    atomic_json(out / "result.json", result)
    return result
