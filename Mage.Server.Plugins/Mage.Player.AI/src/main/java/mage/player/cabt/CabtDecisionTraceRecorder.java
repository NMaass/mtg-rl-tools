package mage.player.cabt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * CABT bridge: records the PENDING → SELECTED → APPLIED lifecycle of every
 * decision the bridge surfaces, so a game can be debugged and replayed
 * decision by decision.
 */
public final class CabtDecisionTraceRecorder {

    private final List<CabtDecisionTrace> traces = new ArrayList<CabtDecisionTrace>();

    public CabtDecisionTrace recordPending(String method, PendingDecision decision) {
        if (decision == null) {
            throw new IllegalArgumentException("cannot trace a null decision");
        }
        CabtDecisionTrace trace = new CabtDecisionTrace(method, decision.selectType(), decision);
        traces.add(trace);
        return trace;
    }

    public void recordSelected(String traceId, Selection selection) {
        find(traceId).markSelected(selection);
    }

    public void recordApplied(String traceId) {
        find(traceId).markApplied();
    }

    public List<CabtDecisionTrace> getTraces() {
        return Collections.unmodifiableList(new ArrayList<CabtDecisionTrace>(traces));
    }

    public CabtDecisionTrace getLastTrace() {
        return traces.isEmpty() ? null : traces.get(traces.size() - 1);
    }

    private CabtDecisionTrace find(String traceId) {
        for (CabtDecisionTrace trace : traces) {
            if (trace.getTraceId().equals(traceId)) {
                return trace;
            }
        }
        throw new IllegalArgumentException("unknown trace id: " + traceId);
    }
}
