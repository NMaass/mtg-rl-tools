package mage.player.cabt;

import mage.abilities.ActivatedAbility;
import mage.game.Game;

/**
 * CABT bridge: applies a validated PRIORITY selection. PASS_PRIORITY passes
 * priority through Player.pass; every playable option dispatches through
 * PlayerImpl.activateAbility, the engine's own root dispatch (playLand for
 * lands, cast for spells, specialAction/playManaAbility/playAbility for the
 * rest). Returns true when the player responded with an action — also when
 * the activation itself failed or was cancelled mid-flow, matching
 * HumanPlayer: the response is consumed and the engine re-offers priority.
 */
public final class CabtPrioritySelectionApplier {

    public boolean apply(CabtBridgePlayer player, Game game,
                         Selection selection, CabtPriorityPrompt prompt) {
        MagicOption option = prompt.getDecision().options().get(selection.indices().get(0));
        if (option.type() == MagicOptionType.PASS_PRIORITY) {
            player.pass(game);
            return false;
        }
        Object playableIndex = option.payload().get("playableIndex");
        if (!(playableIndex instanceof Integer)) {
            throw new IllegalStateException("priority option " + option + " has no playableIndex payload");
        }
        ActivatedAbility ability = prompt.playableAt((Integer) playableIndex);
        player.activateAbility(ability, game);
        return true;
    }
}
