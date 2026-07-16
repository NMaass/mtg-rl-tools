# Trust-audit promotion checklist

A model checkpoint is eligible for a published comparison only when the exact
`magic-cabt-training-audit` report archived with the experiment establishes:

- raw selected actions refer to legal recorded options;
- model observations contain no oracle-label or private-state fields;
- hidden-card perturbations leave visibility-safe model inputs unchanged;
- recurrent source streams contain complete, contiguous games;
- train and evaluation game identities are recorded and disjoint;
- checkpoint tensors and required held-out metrics are finite;
- the checkpoint loads under PyTorch's restricted `weights_only=True` loader;
- available source files match recorded SHA-256 values;
- belief checkpoints contain vocabulary provenance and calibration metrics;
- stochastic dynamics checkpoints contain open-loop, uncertainty, and collapse
  diagnostics.

Warnings must be explained in the experiment report. Use `--strict` for release
and benchmark gates so unresolved warnings become failures.
