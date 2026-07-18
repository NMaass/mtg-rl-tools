# mtg-state-contract

A source-neutral symbolic state layer for MTGO video, MTGO native logs, MTG
Arena logs, XMage, replay analysis, and model inputs.

The comparator operates **before** tensorization.  Model tensors are lossy and
must not be used to establish rules-state equality.  After validation, use
`MagicCabtTensorizerAdapter` to send the same canonical state through the
existing `VisibilitySafeTensorizer` in `mtg-rl-tools`.

```python
from mtg_state_contract import CanonicalStateFormatter, compare_states

formatter = CanonicalStateFormatter()
xmage = formatter.format(xmage_payload, source="xmage")
video = formatter.format(perceived_payload, source="mtgo-video")
report = compare_states(xmage, video)
```

Comparison is confidence-aware.  Unknown or low-confidence video fields are
reported as `UNKNOWN`, not silently treated as correct or automatically treated
as mismatches.
