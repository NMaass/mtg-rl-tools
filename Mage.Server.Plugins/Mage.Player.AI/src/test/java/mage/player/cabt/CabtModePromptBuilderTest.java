package mage.player.cabt;

import mage.abilities.Mode;
import mage.abilities.Modes;
import mage.abilities.effects.common.DrawCardSourceControllerEffect;
import mage.abilities.effects.common.GainLifeEffect;
import mage.game.Game;
import mage.players.Player;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Task 9: MODE prompts are built from real Modes/Mode objects via
 * modes.getAvailableModes(source, game), with readable effect text.
 */
class CabtModePromptBuilderTest {

    private final CabtModePromptBuilder builder = new CabtModePromptBuilder();

    private final UUID aliceId = UUID.randomUUID();

    private Player alice;
    private Game game;

    private void setUpGame() {
        alice = StubGames.player(aliceId, "Alice", 20, 7);
        LinkedHashMap<UUID, Player> players = new LinkedHashMap<UUID, Player>();
        players.put(aliceId, alice);
        game = StubGames.game(players, aliceId, aliceId);
    }

    /**
     * A two-mode modal ability: "draw a card" or "you gain 3 life". The
     * engine's Modes.choose(...) clears the selection before prompting, so
     * the fixture does the same.
     */
    static Modes twoModes() {
        Modes modes = new Modes();
        modes.getMode().getEffects().add(new DrawCardSourceControllerEffect(1));
        modes.addMode(new Mode(new GainLifeEffect(3)));
        modes.clearSelectedModes();
        return modes;
    }

    @Test
    void modePromptIncludesAvailableModes() {
        setUpGame();
        Modes modes = twoModes();

        PendingDecision decision = builder.build(alice, game, modes, StubGames.ability());

        assertThat(decision.selectType()).isEqualTo(MagicSelectType.MODE);
        assertThat(decision.minCount()).isEqualTo(1);
        assertThat(decision.maxCount()).isEqualTo(1);
        assertThat(decision.options()).hasSize(2);

        List<String> modeIds = new ArrayList<String>();
        for (MagicOption option : decision.options()) {
            assertThat(option.type()).isEqualTo(MagicOptionType.PROMPT_MODE);
            assertThat(option.label()).startsWith("Choose mode: ");
            modeIds.add((String) option.payload().get("modeId"));
        }
        for (Mode mode : modes.values()) {
            assertThat(modeIds).contains(mode.getId().toString());
        }
        // readable rule text, not object identity
        assertThat(((String) decision.options().get(0).payload().get("modeText")).toLowerCase())
                .contains("draw a card");
        assertThat(((String) decision.options().get(1).payload().get("modeText")).toLowerCase())
                .contains("gain 3 life");
    }

    @Test
    void alreadySelectedModesAreExcluded() {
        setUpGame();
        Modes modes = twoModes();
        UUID firstModeId = modes.values().iterator().next().getId();
        modes.addSelectedMode(firstModeId);

        PendingDecision decision = builder.build(alice, game, modes, StubGames.ability());

        assertThat(decision.options()).hasSize(1);
        assertThat(decision.options().get(0).payload().get("modeId"))
                .isNotEqualTo(firstModeId.toString());
    }
}
