package mage.player.cabt;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CabtActionIdsTest {

    @Test
    void sameSemanticOptionsFailClosedInsteadOfGettingOrderBasedReplayIds() {
        MagicOption first = new MagicOption(
                MagicOptionType.PROMPT_OBJECT, "Choose token",
                Collections.<String, Object>singletonMap("targetName", "Goblin"));
        MagicOption second = new MagicOption(
                MagicOptionType.PROMPT_OBJECT, "Choose token",
                Collections.<String, Object>singletonMap("targetName", "Goblin"));

        assertThatThrownBy(() -> CabtActionIds.forOptions(Arrays.asList(first, second)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("AMBIGUOUS_SEMANTIC_ACTION");
    }

    @Test
    void stableSemanticPayloadProducesStableActionIds() {
        MagicOption pass = MagicOptionFactory.passPriority();
        MagicOption target = new MagicOption(
                MagicOptionType.PROMPT_OBJECT, "Target Grizzly Bears",
                Collections.<String, Object>singletonMap("targetRef", "P0:D00007"));

        List<String> first = CabtActionIds.forOptions(Arrays.asList(pass, target));
        List<String> second = CabtActionIds.forOptions(Arrays.asList(pass, target));

        assertThat(second).isEqualTo(first);
        assertThat(first).allMatch(id -> id.matches("a_[0-9a-f]{64}"));
        assertThat(first.get(0)).isNotEqualTo(first.get(1));
    }
}
