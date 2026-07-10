import math
import os
import tempfile
import unittest

from magic_cabt.analysis import (AnalysisCache, AnalysisWorker,
                                 analysis_cache_key, decision_fingerprint,
                                 format_analysis, make_analysis_record)

MODEL_INFO = {"modelId": "fake-model", "checkpointId": "sha256:abc"}


def record(chosen=0, instance=101, timestamp="2026-07-10T00:00:00Z"):
    return {
        "gameId": "game-1",
        "sequenceNumber": 4,
        "selectedIndices": [chosen],
        "timestamp": timestamp,
        "observation": {
            "current": {"turnNumber": 2, "localSeat": 1,
                        "players": [{"seat": 1, "life": 20}]},
            "select": {"type": "TARGET", "minCount": 1, "maxCount": 1,
                       "option": [
                           {"index": 0, "type": "TARGET", "label": "Pass",
                            "payload": {}},
                           {"index": 1, "type": "TARGET",
                            "label": "target token",
                            "payload": {"targetInstanceId": instance,
                                        "canonicalKey": "tok"}},
                           {"index": 2, "type": "TARGET", "label": "Other",
                            "payload": {"canonicalKey": "other"}},
                       ]},
        },
    }


class FingerprintTest(unittest.TestCase):
    def test_stable_across_ids_and_timestamps(self):
        self.assertEqual(
            decision_fingerprint(record(instance=101, timestamp="t1")),
            decision_fingerprint(record(instance=202, timestamp="t2")))

    def test_differs_for_different_situations(self):
        other = record()
        other["observation"]["current"]["turnNumber"] = 9
        self.assertNotEqual(decision_fingerprint(record()),
                            decision_fingerprint(other))

    def test_cache_key_includes_checkpoint(self):
        first = analysis_cache_key(record(), MODEL_INFO)
        second = analysis_cache_key(
            record(), {"modelId": "fake-model", "checkpointId": "sha256:zzz"})
        self.assertNotEqual(first, second)


class MakeAnalysisRecordTest(unittest.TestCase):
    def test_ranking_and_probabilities(self):
        result = make_analysis_record(
            record(chosen=0), [2.0, 1.0, 3.0], MODEL_INFO, top_k=5)
        top = result["analysis"]["topK"]
        self.assertEqual([2, 0, 1], [row["optionIndex"] for row in top])
        self.assertAlmostEqual(
            1.0, sum(row["probability"] for row in top), places=6)
        self.assertEqual(2, result["analysis"]["chosenRank"])
        self.assertEqual("other", top[0]["canonicalGroupKey"])
        self.assertEqual(1, result["schemaVersion"])
        self.assertEqual(
            AnalysisCache.key_of(result),
            analysis_cache_key(record(), MODEL_INFO))

    def test_score_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            make_analysis_record(record(), [1.0], MODEL_INFO)

    def test_non_finite_scores_rejected(self):
        with self.assertRaises(ValueError):
            make_analysis_record(
                record(), [1.0, float("nan"), 0.0], MODEL_INFO)

    def test_format_marks_untrained(self):
        result = make_analysis_record(
            record(), [0.0, 1.0, 2.0],
            dict(MODEL_INFO, trainingState="untrained"))
        text = format_analysis(result)
        self.assertIn("(untrained)", text)
        self.assertIn("Played rank:", text)


class AnalysisCacheTest(unittest.TestCase):
    def test_persist_and_reload(self):
        result = make_analysis_record(record(), [0.0, 1.0, 2.0], MODEL_INFO)
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "analysis.jsonl")
            AnalysisCache(path).add(result, persist=True)
            reloaded = AnalysisCache(path)
            self.assertEqual(
                result["analysisId"],
                reloaded.get(AnalysisCache.key_of(result))["analysisId"])

    def test_record_without_identity_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisCache().add({"analysis": {}})


class FakeScorer(object):
    def __init__(self):
        self.calls = 0
        self.model_info = MODEL_INFO

    def score(self, item):
        self.calls += 1
        options = ((item.get("observation") or {}).get("select") or {}).get(
            "option") or []
        return [float(index) for index in range(len(options))]

    def state_value(self, item):
        return 0.25


class AnalysisWorkerTest(unittest.TestCase):
    def test_scores_and_caches(self):
        scorer = FakeScorer()
        cache = AnalysisCache()
        results = []
        worker = AnalysisWorker(scorer, cache=cache, top_k=2,
                                callback=results.append)
        try:
            self.assertTrue(worker.submit(record()))
            self.assertTrue(worker.submit(record()))
        finally:
            worker.close(drain=True, timeout=10.0)
        self.assertEqual(2, len(results))
        self.assertEqual(1, scorer.calls)  # second submit hit the cache
        self.assertEqual(2, len(results[0]["analysis"]["topK"]))
        self.assertEqual(0.25, results[0]["analysis"]["value"])
        self.assertEqual("live", results[0]["source"])

    def test_scorer_errors_reported_not_raised(self):
        class Exploding(FakeScorer):
            def score(self, item):
                raise RuntimeError("boom")

        errors = []
        worker = AnalysisWorker(Exploding(), error_callback=errors.append)
        try:
            worker.submit(record())
        finally:
            worker.close(drain=True, timeout=10.0)
        self.assertEqual(1, len(errors))
        self.assertIn("boom", errors[0])


if __name__ == "__main__":
    unittest.main()
