"""Replay tooling: annotate recorded decisions with a policy.

``annotate`` scores each decision in a decision-record stream with a policy
and writes ``annotations.jsonl`` (top-K options + the chosen option's rank),
model-agnostic so Agent 2's model scorer can later plug in.
"""
