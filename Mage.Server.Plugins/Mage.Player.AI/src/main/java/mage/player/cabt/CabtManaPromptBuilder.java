package mage.player.cabt;

import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.costs.mana.ManaCost;
import mage.abilities.mana.ActivatedManaAbilityImpl;
import mage.constants.ManaType;
import mage.game.Game;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * CABT bridge: builds the PAY_MANA prompt for one step of XMage's payment
 * loop — the engine calls playMana repeatedly while the cost is unpaid, so
 * each prompt picks exactly one mana source, pool mana, or cancel.
 * <p>
 * Discovery reuses the inherited player's own mana path (the protected
 * PlayerImpl helpers getAvailableManaProducers and getUseableManaAbilities,
 * exposed package-privately by CabtBridgePlayer), plus the public
 * ManaPool.get(ManaType) for mana already floating. Special mana actions are
 * not discovered yet; the option kind exists so they can be added without a
 * schema change.
 */
public final class CabtManaPromptBuilder {

    public PendingDecision build(CabtBridgePlayer player, Game game, Ability abilityToCast,
                                 ManaCost unpaid, String promptText) {
        PendingDecision decision = new PendingDecision(
                MagicSelectType.PAY_MANA, player.getId(), 1, 1);
        List<MagicOption> payable = new ArrayList<MagicOption>();
        for (MageObject producer : player.cabtManaProducers(game)) {
            for (ActivatedManaAbilityImpl manaAbility : player.cabtManaAbilities(producer, game).values()) {
                payable.add(CabtManaOptionFactory.manaSourceOption(
                        game, producer, manaAbility, abilityToCast, unpaid, promptText));
            }
        }
        for (ManaType manaType : ManaType.values()) {
            int available = player.getManaPool().get(manaType);
            if (available > 0) {
                payable.add(CabtManaOptionFactory.manaPoolOption(
                        game, manaType, available, abilityToCast, unpaid, promptText));
            }
        }
        payable.sort(new Comparator<MagicOption>() {
            @Override
            public int compare(MagicOption left, MagicOption right) {
                int semantic = CabtSemanticOrder.optionKey(left)
                        .compareTo(CabtSemanticOrder.optionKey(right));
                if (semantic != 0) {
                    return semantic;
                }
                return String.valueOf(left.payload().get("objectId"))
                        .compareTo(String.valueOf(right.payload().get("objectId")));
            }
        });
        for (MagicOption option : payable) {
            decision.addOption(option);
        }
        // Cancel stays deliberately last and is not part of collection-order
        // canonicalization.
        decision.addOption(CabtManaOptionFactory.cancelOption(game, abilityToCast, unpaid, promptText));
        return decision;
    }
}
