package mage.player.cabt;

import mage.game.Game;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.WeakHashMap;

/**
 * Stable protocol identities for runtime XMage objects whose UUIDs are
 * intentionally process-local.
 *
 * Initial players and deck cards are registered from declared input order.
 * The registry is verification/protocol metadata only; XMage continues using
 * its own UUIDs internally.
 */
public final class CabtSemanticIds {

    private static final Map<Game, Map<UUID, String>> IDS =
            new WeakHashMap<Game, Map<UUID, String>>();

    private CabtSemanticIds() {
    }

    public static synchronized void register(Game game, UUID id, String semanticId) {
        if (game == null || id == null || semanticId == null || semanticId.isEmpty()) {
            throw new IllegalArgumentException("game, runtime id, and semantic id are required");
        }
        Map<UUID, String> ids = IDS.get(game);
        if (ids == null) {
            ids = new HashMap<UUID, String>();
            IDS.put(game, ids);
        }
        String previous = ids.put(id, semanticId);
        if (previous != null && !previous.equals(semanticId)) {
            ids.put(id, previous);
            throw new IllegalStateException(
                    "runtime object already has semantic id " + previous);
        }
    }

    public static synchronized String get(Game game, UUID id) {
        Map<UUID, String> ids = IDS.get(game);
        return ids == null || id == null ? null : ids.get(id);
    }

    public static synchronized void clear(Game game) {
        IDS.remove(game);
    }
}
