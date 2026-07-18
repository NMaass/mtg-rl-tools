# xmage-state-follower

Replays semantic actions inferred from MTGO video through the existing
`CabtProtocolServer`, formats both video and engine states through
`mtg-state-contract`, and emits structured mismatch reports.

## Important replay constraint

Exact replay requires:

- both decklists;
- a deterministic bridge seed or an ordered-deck extension;
- the starting player and mulligan choices;
- every engine decision that affects state.

Video frequently omits priority passes and hidden random information. The
follower therefore uses a beam of compatible legal XMage options and labels
ambiguity instead of inventing a single action. The current bridge has no cheap
state clone, so each branch is reconstructed by restarting and replaying the
selection prefix. This is suitable for proof-of-method; a future XMage clone
command can replace the backend without changing the follower.

```powershell
xmage-follow `
  --actions runs\match-001\observed_actions.jsonl `
  --states runs\match-001\canonical_states.jsonl `
  --manifest replay-manifest.json `
  --classpath $env:MAGIC_CABT_CLASSPATH `
  --out runs\match-001\xmage_follow_report.json
```

The comparator is symbolic and confidence-aware. It does not compare model
embeddings or require unreadable video fields to match.

A legal replay is not automatically marked verified. Every consumed visible
action must have a nearby perceived canonical state, and every such comparison
must pass. Unknown/unobservable fields are preserved in the report; known
mismatches fail the branch under the configured threshold.
