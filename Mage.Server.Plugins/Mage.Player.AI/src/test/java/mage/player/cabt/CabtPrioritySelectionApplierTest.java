package mage.player.cabt;

import mage.abilities.ActivatedAbility;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.players.Player;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * PASS_PRIORITY passes and reports "no action"; a playable option must carry
 * the playableIndex that links it back to the live ability. The playable
 * dispatch itself (activateAbility into the engine) is covered by the
 * real-engine smoke tests — stub games cannot host it honestly.
 */
class CabtPrioritySelectionApplierTest {

    private final CabtPriorityPromptBuilder builder = new CabtPriorityPromptBuilder();
    private final CabtPrioritySelectionApplier applier = new CabtPrioritySelectionApplier();

    private CabtBridgePlayer player;
    private Game game;

    private void setUp() {
        player = new CabtBridgePlayer("CABT", RangeOfInfluence.ALL,
                new ScriptedBridgeController(Collections.<Selection>emptyList()));
        LinkedHashMap<UUID, Player> players = new LinkedHashMap<UUID, Player>();
        players.put(player.getId(), player);
        game = StubGames.game(players, player.getId(), player.getId());
    }

    @Test
    void passOptionPassesAndReportsNoAction() {
        setUp();
        CabtPriorityPrompt prompt = builder.build(
                player, game, Collections.<ActivatedAbility>emptyList());

        boolean acted = applier.apply(player, game, Selection.of(0), prompt);

        assertThat(acted).isFalse();
        assertThat(player.isPassed()).isTrue();
    }

    @Test
    void playableOptionWithoutIndexPayloadFailsLoudly() {
        setUp();
        PendingDecision decision = PendingDecision.priority(player.getId());
        decision.addOption(new MagicOption(MagicOptionType.PLAY_LAND, "Play Forest",
                Collections.<String, Object>emptyMap()));
        CabtPriorityPrompt prompt = new CabtPriorityPrompt(
                decision, Collections.<ActivatedAbility>emptyList());

        assertThatThrownBy(() -> applier.apply(player, game, Selection.of(1), prompt))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("playableIndex");
        assertThat(player.isPassed()).isFalse();
    }

    @Test
    void staleIndexOutsidePromptFailsLoudly() {
        setUp();
        PendingDecision decision = PendingDecision.priority(player.getId());
        LinkedHashMap<String, Object> payload = new LinkedHashMap<String, Object>();
        payload.put("playableIndex", 0);
        decision.addOption(new MagicOption(MagicOptionType.PLAY_LAND, "Play Forest", payload));
        CabtPriorityPrompt prompt = new CabtPriorityPrompt(
                decision, Collections.<ActivatedAbility>emptyList());

        assertThatThrownBy(() -> applier.apply(player, game, Selection.of(1), prompt))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("playableIndex 0 outside");
    }
}
