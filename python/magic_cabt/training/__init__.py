"""Training and evaluation utilities.

This package will host the canonical DecisionRecord schema (issue #10),
the IL compiler (issue #11), and baseline evaluators. Sub-modules are
imported directly (``from magic_cabt.training.compile_il import ...``);
the package init deliberately stays minimal so that branches landing one
sub-module but not others do not break each other's imports.
"""