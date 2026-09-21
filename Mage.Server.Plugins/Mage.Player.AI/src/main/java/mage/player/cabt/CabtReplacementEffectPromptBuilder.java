package mage.player.cabt;

import mage.MageObject;\nimport mage.game.Game;\nimport mage.players.Player;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * CABT bridge: builds the REPLACEMENT_EFFECT prompt from the effectsMap the
 * engine passes to chooseReplacementEffect, preserving its iteration order —
 * the same key/text source HumanPlayer turns into a Choice. The answer XMage
 * expects is the chosen entry's index in that order.
 */
public final class CabtReplacementEffectPromptBuilder {

    public PendingDecision build(Player player, Map<String, String> effectsMap,
                                 Map<String, MageObject> objectsMap) {
        return build(player, null, effectsMap, objectsMap);
    }

    public PendingDecision build(Player player, Game game, Map<String, String> effectsMap,
                                 Map<String, MageObject> objectsMap) {
        PendingDecision decision = new PendingDecision(
                MagicSelectType.REPLACEMENT_EFFECT, player.getId(), 1, 1);
        List<MagicOption> options = new ArrayList<MagicOption>();
        int originalIndex = 0;
        for (Map.Entry<String, String> entry : effectsMap.entrySet()) {
            MageObject object = objectsMap == null ? null : objectsMap.get(entry.getKey());
            options.add(CabtReplacementEffectOptionFactory.replacementEffectOption(
                    game, entry.getKey(), entry.getValue(), object, originalIndex));
            originalIndex++;
        }
        options.sort(new Comparator<MagicOption>() {
            @Override
            public int compare(MagicOption left, MagicOption right) {
                int semantic = CabtSemanticOrder.optionKey(left)
                        .compareTo(CabtSemanticOrder.optionKey(right));
                if (semantic != 0) {
                    return semantic;
                }
                return Integer.compare(
                        (Integer) left.payload().get("originalIndex"),
                        (Integer) right.payload().get("originalIndex"));
            }
        });
        for (MagicOption option : options) {
            decision.addOption(option);
        }
        return decision;
    }
}
