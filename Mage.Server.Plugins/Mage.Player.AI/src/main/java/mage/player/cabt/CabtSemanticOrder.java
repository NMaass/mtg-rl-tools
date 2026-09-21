package mage.player.cabt;

import mage.MageObject;
import mage.abilities.ActivatedAbility;
import mage.counters.Counter;
import mage.game.Controllable;
import mage.game.Game;
import mage.game.Ownerable;
import mage.game.permanent.Permanent;
import mage.players.Player;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Stable semantic ordering keys for bridge options.
 *
 * XMage intentionally uses UUIDs as object identities. Those UUIDs are not
 * reproducible across fresh JVM game constructions, so UUID ordering must not
 * leak into the agent-facing option index contract. These keys use game
 * semantics first (player seat, zone, card/object name, ownership/controller,
 * and permanent state). Raw UUIDs may only be used by callers as a final
 * within-run tie breaker for semantically indistinguishable objects.
 */
public final class CabtSemanticOrder {

    private CabtSemanticOrder() {
    }

    public static String targetKey(Game game, UUID id) {
        Player player = game.getPlayer(id);
        if (player != null) {
            return "0|player|" + pad(playerIndex(game, id)) + "|" + safe(player.getName());
        }
        MageObject object = game.getObject(id);
        if (object == null) {
            object = game.getCard(id);
        }
        if (object == null) {
            return "9|unresolved";
        }

        StringBuilder key = new StringBuilder();
        key.append("1|object|");
        String semanticId = CabtSemanticIds.get(game, id);
        if (semanticId != null) {
            key.append("ref=").append(semanticId).append('|');
        }
        key.append(game.getState() == null || game.getState().getZone(id) == null
                ? "" : game.getState().getZone(id).name());
        key.append('|').append(safe(object.getName()));
        key.append('|').append(object.getClass().getName());
        key.append("|zcc=").append(object.getZoneChangeCounter(game));
        if (object instanceof Ownerable) {
            key.append("|owner=").append(pad(playerIndex(
                    game, ((Ownerable) object).getOwnerId())));
        }
        if (object instanceof Controllable) {
            key.append("|controller=").append(pad(playerIndex(
                    game, ((Controllable) object).getControllerId())));
        }
        if (object instanceof Permanent) {
            Permanent permanent = (Permanent) object;
            key.append("|tapped=").append(permanent.isTapped());
            key.append("|faceDown=").append(permanent.isFaceDown(game));
            key.append("|power=").append(permanent.getPower() == null
                    ? "" : permanent.getPower().getValue());
            key.append("|toughness=").append(permanent.getToughness() == null
                    ? "" : permanent.getToughness().getValue());
            key.append("|damage=").append(permanent.getDamage());
            key.append("|summoningSickness=").append(permanent.hasSummoningSickness());
            key.append("|attacking=").append(permanent.isAttacking());
            key.append("|blocked=").append(permanent.isBlocked(game));
            key.append("|attachedTo=").append(targetKey(game, permanent.getAttachedTo()));
            List<String> counters = new ArrayList<String>();
            if (permanent.getCounters(game) != null) {
                for (Counter counter : permanent.getCounters(game).values()) {
                    counters.add(counter.getName() + "=" + counter.getCount());
                }
            }
            Collections.sort(counters);
            key.append("|counters=").append(counters);
        }
        return key.toString();
    }

    public static String abilityKey(Game game, ActivatedAbility ability) {
        StringBuilder key = new StringBuilder();
        key.append(ability.getAbilityType() == null ? "" : ability.getAbilityType().name());
        key.append('|').append(targetKey(game, ability.getSourceId()));
        key.append('|').append(safe(ability.getClass().getName()));
        key.append('|').append(safe(ability.getRule()));
        key.append('|').append(ability.getManaCostsToPay() == null
                ? "" : safe(ability.getManaCostsToPay().getText()));
        return key.toString();
    }

    public static String optionKey(MagicOption option) {
        StringBuilder key = new StringBuilder();
        key.append(option.type() == null ? "" : option.type().name());
        key.append('|').append(safe(option.label()));
        String[] semanticFields = new String[]{
                "sourceRef", "targetRef", "objectRef", "attackerRef", "defenderRef",\n                "blockerRef", "defendingPlayerRef", "sourceName", "rule", "manaCost", "modeText", "selectedCount",
                "objectName", "objectClass", "abilityRule", "manaType", "available",
                "effectText", "choiceValue", "choiceLabel", "targetName",
                "targetClass", "zone", "abilityClass", "unpaid", "promptText"
        };
        for (String field : semanticFields) {
            if (option.payload().containsKey(field)) {
                key.append('|').append(field).append('=')
                        .append(safe(option.payload().get(field)));
            }
        }
        return key.toString();
    }

    public static int playerIndex(Game game, UUID id) {
        if (id == null || game == null) {
            return Integer.MAX_VALUE;
        }
        Player player = game.getPlayer(id);
        if (player instanceof CabtBridgePlayer) {
            int seat = ((CabtBridgePlayer) player).cabtSeat();
            if (seat >= 0) {
                return seat;
            }
        }
        if (game.getPlayerList() == null) {
            return Integer.MAX_VALUE;
        }
        int index = 0;
        for (UUID playerId : game.getPlayerList()) {
            if (id.equals(playerId)) {
                return index;
            }
            index++;
        }
        return Integer.MAX_VALUE;
    }

    private static String pad(int value) {
        return value == Integer.MAX_VALUE ? "x" : String.format("%03d", value);
    }

    private static String safe(Object value) {
        return value == null ? "" : String.valueOf(value);
    }
}
