package mage.player.cabt;

import mage.abilities.TriggeredAbility;
import mage.cards.Card;
import mage.game.Game;
import mage.game.stack.SpellStack;
import mage.players.Player;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * TRIGGER_ORDER prompts carry one option per waiting triggered ability in
 * canonical semantic order, with rule text and source context.
 */
class CabtTriggeredAbilityPromptBuilderTest {

    private final CabtTriggeredAbilityPromptBuilder builder = new CabtTriggeredAbilityPromptBuilder();
    private final CabtTriggeredAbilitySelectionApplier applier = new CabtTriggeredAbilitySelectionApplier();

    private final UUID aliceId = UUID.randomUUID();
    private final UUID bearsId = UUID.randomUUID();
    private final UUID wolfId = UUID.randomUUID();

    private Player alice;
    private Game game;
    private List<TriggeredAbility> triggers;

    private void setUpTwoTriggers() {
        alice = StubGames.player(aliceId, "Alice", 20, 7);
        LinkedHashMap<UUID, Player> players = new LinkedHashMap<UUID, Player>();
        players.put(aliceId, alice);
        LinkedHashMap<UUID, Card> cards = new LinkedHashMap<UUID, Card>();
        cards.put(bearsId, StubGames.card(bearsId, "Grizzly Bears", aliceId));
        cards.put(wolfId, StubGames.card(wolfId, "Young Wolf", aliceId));
        game = StubGames.game(players, aliceId, aliceId, null, new SpellStack(), cards);
        triggers = Arrays.asList(
                StubGames.triggeredAbility(UUID.randomUUID(), UUID.randomUUID(), bearsId,
                        "Whenever Grizzly Bears attacks, draw a card."),
                StubGames.triggeredAbility(UUID.randomUUID(), UUID.randomUUID(), wolfId,
                        "When Young Wolf dies, return it with a +1/+1 counter."));
    }

    @Test
    void triggerPromptIncludesEachAbility() {
        setUpTwoTriggers();

        PendingDecision decision = builder.build(alice, game, triggers);

        assertThat(decision.selectType()).isEqualTo(MagicSelectType.TRIGGER_ORDER);
        assertThat(decision.minCount()).isEqualTo(1);
        assertThat(decision.maxCount()).isEqualTo(1);
        assertThat(decision.options()).hasSize(2);
        for (MagicOption option : decision.options()) {
            assertThat(option.type()).isEqualTo(MagicOptionType.PROMPT_TRIGGERED_ABILITY);
            assertThat(option.label()).startsWith("Put trigger on stack: ");
            String abilityId = String.valueOf(option.payload().get("abilityId"));
            TriggeredAbility matching = triggers.stream()
                    .filter(trigger -> trigger.getId().toString().equals(abilityId))
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("option does not map to an engine trigger"));
            assertThat(option.payload().get("sourceId"))
                    .isEqualTo(matching.getSourceId().toString());
            assertThat(option.payload().get("rule")).isNotNull();
        }
        assertThat(decision.options().stream()
                .map(option -> option.payload().get("sourceName"))
                .collect(Collectors.toList()))
                .containsExactly("Grizzly Bears", "Young Wolf");
    }

    @Test
    void triggerOrderIsSemanticRatherThanInputListOrder() {
        setUpTwoTriggers();
        PendingDecision forward = builder.build(alice, game, triggers);
        List<TriggeredAbility> reversed = new ArrayList<TriggeredAbility>(triggers);
        Collections.reverse(reversed);
        PendingDecision backward = builder.build(alice, game, reversed);

        assertThat(backward.options().stream().map(MagicOption::label).collect(Collectors.toList()))
                .isEqualTo(forward.options().stream().map(MagicOption::label).collect(Collectors.toList()));
        TriggeredAbility picked = applier.apply(reversed, Selection.of(0), backward);
        assertThat(picked.getId().toString())
                .isEqualTo(backward.options().get(0).payload().get("abilityId"));
    }

    @Test
    void triggerSelectionReturnsSelectedAbility() {
        setUpTwoTriggers();
        PendingDecision decision = builder.build(alice, game, triggers);

        TriggeredAbility expected = triggers.get(1);
        int optionIndex = -1;
        for (int i = 0; i < decision.options().size(); i++) {
            if (expected.getId().toString().equals(
                    decision.options().get(i).payload().get("abilityId"))) {
                optionIndex = i;
                break;
            }
        }
        assertThat(optionIndex).isGreaterThanOrEqualTo(0);
        TriggeredAbility selected = applier.apply(triggers, Selection.of(optionIndex), decision);

        assertThat(selected).isSameAs(expected);
    }
}
