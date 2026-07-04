package mage.player.cabt;

import java.util.Collections;
import java.util.Map;

/**
 * CABT bridge: factories for well-known options, so option objects are built in one
 * place instead of ad hoc at every decision point. Task 1 added passPriority();
 * target prompts add the PROMPT_* answers; later tasks add playLand, castSpell, etc.
 */
public final class MagicOptionFactory {

    private MagicOptionFactory() {
    }

    public static MagicOption passPriority() {
        return new MagicOption(
                MagicOptionType.PASS_PRIORITY,
                "Pass priority",
                Collections.<String, Object>emptyMap()
        );
    }

    public static MagicOption promptPlayer(String label, Map<String, Object> payload) {
        return new MagicOption(MagicOptionType.PROMPT_PLAYER, label, payload);
    }

    public static MagicOption promptObject(String label, Map<String, Object> payload) {
        return new MagicOption(MagicOptionType.PROMPT_OBJECT, label, payload);
    }

    public static MagicOption promptCard(String label, Map<String, Object> payload) {
        return new MagicOption(MagicOptionType.PROMPT_CARD, label, payload);
    }
}
