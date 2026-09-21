package mage.player.cabt;

import mage.abilities.TriggeredAbility;
import mage.game.Game;
import mage.players.Player;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * CABT bridge: builds the TRIGGER_ORDER prompt when GameImpl.checkTriggered()
 * asks which waiting triggered ability goes on the stack first. Option index
 * i corresponds to abilities.get(i) — the input list's order is the prompt's
 * order.
 */
public final class CabtTriggeredAbilityPromptBuilder {

    public PendingDecision build(Player player, Game game, List<TriggeredAbility> abilities) {
        PendingDecision decision = new PendingDecision(
                MagicSelectType.TRIGGER_ORDER, player.getId(), 1, 1);
        List<MagicOption> options = new ArrayList<MagicOption>();
        for (TriggeredAbility ability : abilities) {
            options.add(CabtTriggeredAbilityOptionFactory.triggeredAbilityOption(game, ability));
        }
        options.sort(new Comparator<MagicOption>() {
            @Override
            public int compare(MagicOption left, MagicOption right) {
                int semantic = CabtSemanticOrder.optionKey(left)
                        .compareTo(CabtSemanticOrder.optionKey(right));
                if (semantic != 0) {
                    return semantic;
                }
                return String.valueOf(left.payload().get("abilityId"))
                        .compareTo(String.valueOf(right.payload().get("abilityId")));
            }
        });
        for (MagicOption option : options) {
            decision.addOption(option);
        }
        return decision;
    }
}
