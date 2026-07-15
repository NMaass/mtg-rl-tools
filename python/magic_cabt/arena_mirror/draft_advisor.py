"""Live draft advice for the mirror: pack scores and a deck outlook.

Loads a trained card-selection checkpoint (``magic-cabt-train-draft``) and
answers two questions during a running draft:

- ``score_pack``: how does the model rank the cards in the pack on offer,
  given the pool drafted so far;
- ``outlook``: what deck does the pool greedily build right now.

Entirely optional: ``DraftAdvisor.maybe_load`` returns None when torch or the
checkpoint is absent, and the mirror's draft pane then shows names without
scores. The checkpoint path comes from ``MAGIC_CABT_DRAFT_CHECKPOINT`` or the
default ``~/.magic-cabt/draft-model/checkpoint.pt``.
"""

import os

__all__ = ["DraftAdvisor", "default_draft_checkpoint_path"]

CHECKPOINT_ENV_VAR = "MAGIC_CABT_DRAFT_CHECKPOINT"


def default_draft_checkpoint_path():
    return os.environ.get(CHECKPOINT_ENV_VAR) or os.path.join(
        os.path.expanduser("~"), ".magic-cabt", "draft-model",
        "checkpoint.pt")


class DraftAdvisor(object):

    def __init__(self, model, tensorizer):
        self.model = model
        self.tensorizer = tensorizer
        self.resolver = tensorizer.resolver

    @classmethod
    def maybe_load(cls, card_db=None, checkpoint_path=None, log=None):
        """Load the advisor if torch and a checkpoint exist, else None."""
        path = checkpoint_path or default_draft_checkpoint_path()
        if not os.path.isfile(path):
            return None
        try:
            from ..models.draft_model import (CardSelectionModel,
                                              DraftCardResolver,
                                              DraftTensorizer)
            from ..models.structured_config import CardTextResolver
            from .cards import default_mtga_card_db_path
        except ImportError as error:
            if log:
                log("draft advisor unavailable: %s" % (error,))
            return None
        try:
            model, _extra = CardSelectionModel.load_checkpoint(path)
            model.eval()
            resolver = DraftCardResolver(card_db, CardTextResolver(
                arena_db_path=default_mtga_card_db_path()))
            tensorizer = DraftTensorizer(model.config, resolver)
        except Exception as error:
            if log:
                log("draft advisor failed to load %s: %s" % (path, error))
            return None
        if log:
            log("draft advisor loaded: %s" % path)
        return cls(model, tensorizer)

    def name_for(self, grp_id):
        return self.resolver.describe(grp_id).get("name") or str(grp_id)

    def score_pack(self, draft):
        """Ranked ``[{grpId, name, score}]`` for the pack in ``draft``."""
        import torch

        pack = list(draft.get("packCards") or [])
        if not pack:
            return []
        example = {
            "mode": "draftPick",
            "contextIds": list(draft.get("pool") or []),
            "candidateIds": pack,
            "packNumber": draft.get("packNumber"),
            "pickNumber": draft.get("pickNumber"),
        }
        with torch.no_grad():
            tensors = self.tensorizer.batch_examples([example])
            logits = self.model.score_candidates(*tensors)[0]
        scored = [{"grpId": grp_id, "name": self.name_for(grp_id),
                   "score": float(logits[index])}
                  for index, grp_id in enumerate(pack)]
        scored.sort(key=lambda item: -item["score"])
        return scored

    def outlook(self, pool):
        """Greedy deck build of ``pool``; see analysis.draft_outlook."""
        from ..analysis.draft_outlook import deck_outlook

        if not pool:
            return None
        return deck_outlook(self.model, self.tensorizer, list(pool))
