package mage.player.cabt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * CABT bridge: a selection prompt built at an XMage player-input callback.
 * Equivalent of CABT's setSelect + addOption: select type, selecting player,
 * min/max selection count, and an ordered list of options addressed by index.
 */
public final class PendingDecision {

    private final MagicSelectType selectType;
    private final UUID playerId;
    private final int minCount;
    private final int maxCount;
    private final List<MagicOption> options = new ArrayList<>();

    // package-private: decisions are built through the static factories; tests may
    // construct arbitrary min/max prompts directly
    PendingDecision(MagicSelectType selectType, UUID playerId, int minCount, int maxCount) {
        this.selectType = selectType;
        this.playerId = playerId;
        this.minCount = minCount;
        this.maxCount = maxCount;
    }

    /**
     * Prompt for a priority stop: choose exactly one action.
     * Task 1 offers PASS_PRIORITY only; later tasks add the playable actions.
     */
    public static PendingDecision priority(UUID playerId) {
        PendingDecision decision = new PendingDecision(MagicSelectType.PRIORITY, playerId, 1, 1);
        decision.addOption(MagicOptionFactory.passPriority());
        return decision;
    }

    public void addOption(MagicOption option) {
        options.add(option);
    }

    public MagicSelectType selectType() {
        return selectType;
    }

    public UUID playerId() {
        return playerId;
    }

    public int minCount() {
        return minCount;
    }

    public int maxCount() {
        return maxCount;
    }

    public List<MagicOption> options() {
        return Collections.unmodifiableList(options);
    }
}
