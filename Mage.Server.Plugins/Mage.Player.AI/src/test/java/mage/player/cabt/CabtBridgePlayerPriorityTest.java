package mage.player.cabt;

import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.players.Player;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Proves the Task 1 seam: priority() builds a CABT-style prompt, asks the bridge,
 * validates the answer, and dispatches PASS_PRIORITY as pass(game) + return false —
 * matching ComputerPlayer's minimum priority behavior, but routed through the bridge.
 */
class CabtBridgePlayerPriorityTest {

    /**
     * Records every prompt it is asked, then replays scripted selections.
     */
    private static final class PromptRecordingBridge implements CabtBridgeController {

        private final ScriptedBridgeController script;
        private final List<PendingDecision> requestedDecisions = new ArrayList<>();

        private PromptRecordingBridge(Selection... selections) {
            List<Selection> list = new ArrayList<>();
            Collections.addAll(list, selections);
            this.script = new ScriptedBridgeController(list);
        }

        @Override
        public Selection requestSelection(Game game, Player player, PendingDecision decision) {
            requestedDecisions.add(decision);
            return script.requestSelection(game, player, decision);
        }
    }

    /**
     * Minimal Game stub: pass(game) does not need any game state when no bookmark
     * is stored, so every method just answers a type-correct default.
     */
    private static Game stubGame() {
        InvocationHandler handler = new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) {
                Class<?> returnType = method.getReturnType();
                if (returnType == boolean.class) {
                    return false;
                }
                if (returnType.isPrimitive() && returnType != void.class) {
                    return 0;
                }
                return null;
            }
        };
        return (Game) Proxy.newProxyInstance(
                Game.class.getClassLoader(), new Class<?>[]{Game.class}, handler);
    }

    @Test
    void priorityRoutesThroughBridgeThenPasses() {
        PromptRecordingBridge bridge = new PromptRecordingBridge(Selection.of(0));
        CabtBridgePlayer player = new CabtBridgePlayer("cabt-1", RangeOfInfluence.ALL, bridge);
        Game game = stubGame();

        assertThat(player.isPassed()).isFalse();

        boolean result = player.priority(game);

        // same contract as ComputerPlayer's minimum implementation
        assertThat(result).isFalse();
        assertThat(player.isPassed()).isTrue();

        // the decision went through the bridge, not an auto-pass
        assertThat(bridge.requestedDecisions).hasSize(1);
        PendingDecision decision = bridge.requestedDecisions.get(0);
        assertThat(decision.selectType()).isEqualTo(MagicSelectType.PRIORITY);
        assertThat(decision.playerId()).isEqualTo(player.getId());
        assertThat(decision.minCount()).isEqualTo(1);
        assertThat(decision.maxCount()).isEqualTo(1);
        assertThat(decision.options()).hasSize(1);
        assertThat(decision.options().get(0).type()).isEqualTo(MagicOptionType.PASS_PRIORITY);
    }

    @Test
    void priorityRejectsInvalidSelectionWithoutPassing() {
        PromptRecordingBridge bridge = new PromptRecordingBridge(Selection.of(1));
        CabtBridgePlayer player = new CabtBridgePlayer("cabt-1", RangeOfInfluence.ALL, bridge);

        assertThatThrownBy(() -> player.priority(stubGame()))
                .isInstanceOf(InvalidSelectionException.class)
                .hasMessage("OPTION_INDEX_OUT_OF_RANGE");

        assertThat(player.isPassed()).isFalse();
    }

    @Test
    void copyKeepsBridgeRouting() {
        PromptRecordingBridge bridge = new PromptRecordingBridge(Selection.of(0));
        CabtBridgePlayer player = new CabtBridgePlayer("cabt-1", RangeOfInfluence.ALL, bridge);

        CabtBridgePlayer copy = player.copy();
        assertThat(copy.priority(stubGame())).isFalse();
        assertThat(copy.isPassed()).isTrue();
        assertThat(bridge.requestedDecisions).hasSize(1);
    }
}
