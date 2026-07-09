"""Dependency-free behavior-cloning option scorer."""

import json
import math
import re

__all__ = ["BagOfWordsBCPolicy"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


class BagOfWordsBCPolicy(object):
    """Naive bag-of-words legal-option scorer for compiled IL examples."""

    schema_version = 1

    def __init__(self, weights=None, metadata=None):
        self.weights = weights or {"bias": 0.0, "optionType": {}, "token": {}, "promptOptionType": {}}
        self.metadata = metadata or {}

    @classmethod
    def train(cls, examples, min_token_count=1):
        counts = {
            "examples": 0,
            "chosen": {"optionType": {}, "token": {}, "promptOptionType": {}},
            "all": {"optionType": {}, "token": {}, "promptOptionType": {}},
            "promptType": {},
            "tokens": {},
        }
        for example in examples:
            option_texts = list(example.get("optionTexts") or [])
            option_types = list(example.get("optionTypes") or [])
            selected = example.get("chosenIndex")
            prompt = str(example.get("promptType") or "UNKNOWN")
            if not isinstance(selected, int) or isinstance(selected, bool):
                continue
            if selected < 0 or selected >= len(option_texts):
                continue
            counts["examples"] += 1
            _inc(counts["promptType"], prompt)
            for index, text in enumerate(option_texts):
                option_type = _option_type_at(option_types, index)
                key = _prompt_option_key(prompt, option_type)
                _inc(counts["all"]["optionType"], option_type)
                _inc(counts["all"]["promptOptionType"], key)
                tokens = _tokens(text)
                for token in tokens:
                    _inc(counts["all"]["token"], token)
                    _inc(counts["tokens"], token)
                if index == selected:
                    _inc(counts["chosen"]["optionType"], option_type)
                    _inc(counts["chosen"]["promptOptionType"], key)
                    for token in tokens:
                        _inc(counts["chosen"]["token"], token)

        weights = {
            "bias": 0.0,
            "optionType": _log_ratios(counts["chosen"]["optionType"], counts["all"]["optionType"]),
            "promptOptionType": _log_ratios(counts["chosen"]["promptOptionType"], counts["all"]["promptOptionType"]),
            "token": _log_ratios(
                _filtered(counts["chosen"]["token"], counts["tokens"], min_token_count),
                _filtered(counts["all"]["token"], counts["tokens"], min_token_count),
            ),
        }
        metadata = {
            "schemaVersion": cls.schema_version,
            "modelType": "bag_of_words_bc",
            "examples": counts["examples"],
            "minTokenCount": min_token_count,
            "promptTypeCounts": counts["promptType"],
        }
        return cls(weights=weights, metadata=metadata)

    def score_option(self, prompt_type, option_type, option_text):
        prompt_type = str(prompt_type or "UNKNOWN")
        option_type = str(option_type or "UNKNOWN")
        score = float(self.weights.get("bias", 0.0))
        score += self.weights.get("optionType", {}).get(option_type, 0.0)
        score += self.weights.get("promptOptionType", {}).get(_prompt_option_key(prompt_type, option_type), 0.0)
        for token in _tokens(option_text):
            score += self.weights.get("token", {}).get(token, 0.0)
        return score

    def score_example(self, example):
        prompt_type = example.get("promptType") or "UNKNOWN"
        option_texts = list(example.get("optionTexts") or [])
        option_types = list(example.get("optionTypes") or [])
        return [
            self.score_option(prompt_type, _option_type_at(option_types, index), text)
            for index, text in enumerate(option_texts)
        ]

    def rank_example(self, example):
        scores = self.score_example(example)
        return sorted(range(len(scores)), key=lambda index: (-scores[index], index))

    def to_dict(self):
        return {"schemaVersion": self.schema_version, "modelType": "bag_of_words_bc", "weights": self.weights, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("modelType") != "bag_of_words_bc":
            raise ValueError("not a bag_of_words_bc checkpoint")
        return cls(weights=payload.get("weights") or {}, metadata=payload.get("metadata") or {})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _tokens(text):
    return [token.lower() for token in _TOKEN_RE.findall(str(text or ""))]


def _inc(bucket, key):
    key = str(key)
    bucket[key] = bucket.get(key, 0) + 1


def _option_type_at(option_types, index):
    if index < len(option_types) and option_types[index] is not None:
        return str(option_types[index])
    return "UNKNOWN"


def _prompt_option_key(prompt, option_type):
    return "%s/%s" % (prompt, option_type)


def _filtered(counts, token_counts, min_count):
    return {key: value for key, value in counts.items() if token_counts.get(key, 0) >= min_count}


def _log_ratios(chosen, all_options):
    keys = set(chosen) | set(all_options)
    total_chosen = float(sum(chosen.values()))
    total_all = float(sum(all_options.values()))
    vocab = float(max(1, len(keys)))
    weights = {}
    for key in sorted(keys):
        selected_rate = (chosen.get(key, 0) + 1.0) / (total_chosen + vocab)
        available_rate = (all_options.get(key, 0) + 1.0) / (total_all + vocab)
        weights[key] = math.log(selected_rate / available_rate)
    return weights
