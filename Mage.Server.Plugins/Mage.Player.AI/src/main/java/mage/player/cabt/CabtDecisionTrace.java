package mage.player.cabt;

import java.util.UUID;

/**
 * CABT bridge: lifecycle record of one decision — the prompt was built
 * (PENDING), the controller answered (SELECTED), and the answer was applied
 * to the engine (APPLIED). Stages advance strictly in that order.
 * <p>
 * {@code method} names the XMage callback that raised the prompt (e.g.
 * CHOOSE_TARGET, CHOOSE_USE, ANNOUNCE_X); the full prompt — options,
 * min/max, payloads such as number ranges and messages — stays reachable
 * through {@code decision}.
 */
public final class CabtDecisionTrace {

    public enum Stage {
        PENDING,
        SELECTED,
        APPLIED
    }

    private final String traceId;
    private final String method;
    private final MagicSelectType selectType;
    private final PendingDecision decision;
    private Stage stage;
    private Selection selection;

    CabtDecisionTrace(String method, MagicSelectType selectType, PendingDecision decision) {
        this.traceId = UUID.randomUUID().toString();
        this.method = method;
        this.selectType = selectType;
        this.decision = decision;
        this.stage = Stage.PENDING;
    }

    void markSelected(Selection selection) {
        if (stage != Stage.PENDING) {
            throw new IllegalStateException("trace " + traceId + " already " + stage);
        }
        this.stage = Stage.SELECTED;
        this.selection = selection;
    }

    void markApplied() {
        if (stage != Stage.SELECTED) {
            throw new IllegalStateException("trace " + traceId + " is " + stage + ", not SELECTED");
        }
        this.stage = Stage.APPLIED;
    }

    public String getTraceId() {
        return traceId;
    }

    public String getMethod() {
        return method;
    }

    public MagicSelectType getSelectType() {
        return selectType;
    }

    public PendingDecision getDecision() {
        return decision;
    }

    public Stage getStage() {
        return stage;
    }

    public Selection getSelection() {
        return selection;
    }
}
