package mage.player.cabt;

import mage.MageObject;
import mage.abilities.ActivatedAbility;
import mage.game.Game;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

/**
 * CABT bridge: builds one PRIORITY option per playable ability returned by
 * Player.getPlayable — a land play, a castable spell, an activatable ability,
 * or a special action. Any other ability type coming out of getPlayable is a
 * bug and fails closed instead of being bucketed.
 */
public final class CabtPriorityOptionFactory {

    private CabtPriorityOptionFactory() {
    }

    public static MagicOption playableOption(Game game, ActivatedAbility ability, int playableIndex) {
        MagicOptionType type = optionType(ability);
        String sourceName = sourceName(game, ability);
        Map<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("playableIndex", playableIndex);
        payload.put("abilityType", ability.getAbilityType().name());
        payload.put("abilityId", ability.getId().toString());
        payload.put("sourceId", ability.getSourceId() == null ? null : ability.getSourceId().toString());
        payload.put("sourceName", sourceName);
        payload.put("rule", ability.getRule());
        payload.put("manaCost", ability.getManaCostsToPay().getText());
        return new MagicOption(type, label(type, ability, sourceName), payload);
    }

    static MagicOptionType optionType(ActivatedAbility ability) {
        switch (ability.getAbilityType()) {
            case PLAY_LAND:
                return MagicOptionType.PLAY_LAND;
            case SPELL:
                return MagicOptionType.CAST_SPELL;
            case ACTIVATED_NONMANA:
            case ACTIVATED_MANA:
                return MagicOptionType.ACTIVATE_ABILITY;
            case SPECIAL_ACTION:
            case SPECIAL_MANA_PAYMENT:
                return MagicOptionType.SPECIAL_ACTION;
            default:
                throw new CabtUnhandledDecisionException(
                        "getPlayable returned an ability type with no priority option mapping: "
                                + ability.getAbilityType().name() + " (" + ability.getRule() + ')');
        }
    }

    private static String label(MagicOptionType type, ActivatedAbility ability, String sourceName) {
        String name = sourceName != null ? sourceName : ability.getRule();
        switch (type) {
            case PLAY_LAND:
                return "Play " + name;
            case CAST_SPELL:
                return "Cast " + name;
            case ACTIVATE_ABILITY:
                return name + ": " + ability.getRule();
            case SPECIAL_ACTION:
                return ability.getRule();
            default:
                throw new IllegalStateException("not a priority option type: " + type);
        }
    }

    private static String sourceName(Game game, ActivatedAbility ability) {
        UUID sourceId = ability.getSourceId();
        if (sourceId == null) {
            return null;
        }
        MageObject object = game.getObject(sourceId);
        if (object == null) {
            object = game.getCard(sourceId);
        }
        return object == null ? null : object.getName();
    }
}
