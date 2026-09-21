package mage.player.cabt;

import mage.abilities.ActivatedAbility;
import mage.game.Game;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * CABT bridge: builds the PRIORITY prompt — PASS_PRIORITY at index 0, then
 * one option per playable ability the engine enumerated via
 * Player.getPlayable(Game, true) (land plays, castable spells, activatable
 * abilities, special actions). The caller passes the playables in so stubbed
 * tests can drive the builder directly and the bridge player enumerates once
 * per prompt.
 */
public final class CabtPriorityPromptBuilder {

    public CabtPriorityPrompt build(CabtBridgePlayer player, Game game,
                                    List<ActivatedAbility> playables) {
        PendingDecision decision = PendingDecision.priority(player.getId());
        List<ActivatedAbility> ordered = new ArrayList<ActivatedAbility>(playables);
        ordered.sort(new Comparator<ActivatedAbility>() {
            @Override
            public int compare(ActivatedAbility left, ActivatedAbility right) {
                int semantic = CabtSemanticOrder.abilityKey(game, left)
                        .compareTo(CabtSemanticOrder.abilityKey(game, right));
                if (semantic != 0) {
                    return semantic;
                }
                String leftId = left.getId() == null ? "" : left.getId().toString();
                String rightId = right.getId() == null ? "" : right.getId().toString();
                return leftId.compareTo(rightId);
            }
        });
        List<ActivatedAbility> kept = new ArrayList<ActivatedAbility>();
        for (ActivatedAbility ability : ordered) {
            decision.addOption(CabtPriorityOptionFactory.playableOption(game, ability, kept.size()));
            kept.add(ability);
        }
        return new CabtPriorityPrompt(decision, kept);
    }
}
