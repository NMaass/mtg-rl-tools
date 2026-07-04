package mage.player.cabt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * CABT bridge: records the PENDING → SELECTED → APPLIED (or FAILED) lifecycle
 * of every decision the bridge surfaces, numbering traces sequentially so a
 * game can be debugged and replayed decision by decision. Trace N pairs with
 * the N-th observation captured by a recording controller; the transition
 * dataset ({@link CabtDatasetRecord}) is built from that pairing.
 */
public final class CabtDecisionTraceRecorder {

    private final List<CabtDecisionTrace> traces = new ArrayList<CabtDecisionTrace>();
    private long nextSequenceNumber;

    public CabtDecisionTrace recordPending(String method, PendingDecision decision) {
        if (decision == null) {
            throw new IllegalArgumentException("cannot trace a null decision");
        }
        CabtDecisionTrace trace = new CabtDecisionTrace(nextSequenceNumber++, method, decision);
        traces.add(trace);
        return trace;
    }

    public void recordSelected(String traceId, Selection selection) {
        find(traceId).markSelected(selection);
    }

    public void recordApplied(String traceId) {
        find(traceId).markApplied();
    }

    public void recordFailed(String traceId, Throwable error) {
        find(traceId).markFailed(error == null ? "unknown error"
                : error.getClass().getSimpleName() + ": " + error.getMessage());
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
