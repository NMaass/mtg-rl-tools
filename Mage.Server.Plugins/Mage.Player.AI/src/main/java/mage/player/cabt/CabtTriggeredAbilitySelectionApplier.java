package mage.player.cabt;

import mage.abilities.TriggeredAbility;

import java.util.List;

/**
 * CABT bridge: resolves a validated TRIGGER_ORDER selection back to the
 * TriggeredAbility object XMage expects — option index i is abilities.get(i).
 */
public final class CabtTriggeredAbilitySelectionApplier {

    public TriggeredAbility apply(List<TriggeredAbility> abilities,
                                  Selection selection, PendingDecision decision) {
        int index = selection.indices().get(0);
        Object abilityId = decision.options().get(index).payload().get("abilityId");
        if (!(abilityId instanceof String)) {
            throw new IllegalStateException("trigger option has no abilityId payload");
        }
        for (TriggeredAbility ability : abilities) {
            if (ability.getId() != null && abilityId.equals(ability.getId().toString())) {
                return ability;
            }
        }
        throw new IllegalStateException(
                "selected trigger is no longer present in the engine list");
    }
}
