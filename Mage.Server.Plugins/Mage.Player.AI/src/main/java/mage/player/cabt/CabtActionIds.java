package mage.player.cabt;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Stable identifiers for agent-facing legal actions.
 *
 * Numeric option indices remain the live transport representation, but they
 * are not a replay identity. An action id is derived from the semantic option
 * payload and its occurrence among otherwise identical options. Reconstructed
 * prompts can therefore resolve a recorded semantic choice back to the
 * current engine index and fail closed when the action space changed.
 */
public final class CabtActionIds {

    private CabtActionIds() {
    }

    public static List<String> forOptions(List<MagicOption> options) {
        List<String> ids = new ArrayList<String>();
        Map<String, Integer> occurrences = new HashMap<String, Integer>();
        for (MagicOption option : options) {
            String semantic = CabtSemanticOrder.optionKey(option);
            Integer previous = occurrences.get(semantic);
            int occurrence = previous == null ? 0 : previous.intValue() + 1;
            occurrences.put(semantic, occurrence);
            ids.add("a_" + sha256(semantic + "\u0000" + occurrence));
        }
        return ids;
    }

    public static List<Integer> resolve(PendingDecision decision, List<String> selectedIds) {
        List<String> ids = forOptions(decision.options());
        Map<String, Integer> indexById = new HashMap<String, Integer>();
        for (int index = 0; index < ids.size(); index++) {
            String id = ids.get(index);
            if (indexById.put(id, Integer.valueOf(index)) != null) {
                throw new IllegalStateException("ACTION_ID_COLLISION");
            }
        }
        Set<String> seen = new HashSet<String>();
        List<Integer> indices = new ArrayList<Integer>();
        for (String actionId : selectedIds) {
            if (actionId == null || !seen.add(actionId)) {
                throw new InvalidSelectionException("DUPLICATE_ACTION_ID");
            }
            Integer index = indexById.get(actionId);
            if (index == null) {
                throw new InvalidSelectionException("UNKNOWN_ACTION_ID");
            }
            indices.add(index);
        }
        SelectionValidator.validate(decision, new Selection(indices));
        return indices;
    }

    private static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : bytes) {
                hex.append(String.format("%02x", b & 0xff));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 unavailable", impossible);
        }
    }
}
