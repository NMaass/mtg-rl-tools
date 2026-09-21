package mage.player.cabt;

import mage.MageObject;
import mage.cards.Card;
import mage.game.Game;
import mage.game.command.CommandObject;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Private deterministic-state verifier for replay reconstruction.
 *
 * This digest is deliberately separate from agent observations. It includes
 * hidden hands and ordered libraries so a replay cannot be certified merely
 * because its public board happens to match. Raw UUIDs are excluded because
 * XMage allocates them nondeterministically when objects are constructed.
 */
public final class CabtEngineFingerprint {

    private CabtEngineFingerprint() {
    }

    public static String sha256(Game game) {
        return sha256Text(canonical(game));
    }

    static String canonical(Game game) {
        StringBuilder out = new StringBuilder();
        out.append("turn=").append(game.getTurnNum());
        out.append("|phase=").append(game.getTurnPhaseType());
        out.append("|step=").append(game.getTurnStepType());
        out.append("|active=").append(CabtSemanticOrder.playerIndex(game, game.getActivePlayerId()));
        out.append("|priority=").append(CabtSemanticOrder.playerIndex(game, game.getPriorityPlayerId()));
        out.append("|ended=").append(game.hasEnded());
        out.append("|winner=").append(safe(game.getWinner())).append('\n');

        int seat = 0;
        if (game.getPlayerList() != null) {
            for (UUID playerId : game.getPlayerList()) {
                Player player = game.getPlayer(playerId);
                if (player != null) {
                    out.append("P").append(seat)
                            .append("|name=").append(safe(player.getName()))
                            .append("|life=").append(player.getLife())
                            .append("|passed=").append(player.isPassed())
                            .append("|inGame=").append(player.isInGame())
                            .append("|hand=").append(sortedCardNames(game, player.getHand()))
                            .append("|library=").append(orderedCardNames(game, player.getLibrary().getCardList()))
                            .append("|graveyard=").append(orderedCardNames(game, player.getGraveyard()))
                            .append('\n');
                }
                seat++;
            }
        }

        List<String> battlefield = new ArrayList<String>();
        if (game.getBattlefield() != null) {
            for (Permanent permanent : game.getBattlefield().getAllActivePermanents()) {
                battlefield.add(CabtSemanticOrder.targetKey(game, permanent.getId()));
            }
        }
        Collections.sort(battlefield);
        out.append("battlefield=").append(battlefield).append('\n');

        List<String> stack = new ArrayList<String>();
        if (game.getStack() != null) {
            for (StackObject object : game.getStack()) {
                stack.add(stackKey(game, object));
            }
        }
        out.append("stack=").append(stack).append('\n');

        List<String> exile = new ArrayList<String>();
        if (game.getExile() != null) {
            for (Card card : game.getExile().getAllCards(game)) {
                exile.add(cardKey(game, card));
            }
        }
        Collections.sort(exile);
        out.append("exile=").append(exile).append('\n');

        List<String> command = new ArrayList<String>();
        if (game.getState() != null && game.getState().getCommand() != null) {
            for (CommandObject object : game.getState().getCommand()) {
                command.add(objectKey(game, object));
            }
        }
        Collections.sort(command);
        out.append("command=").append(command).append('\n');
        return out.toString();
    }

    private static List<String> sortedCardNames(Game game, Iterable<UUID> ids) {
        List<String> values = orderedCardNames(game, ids);
        Collections.sort(values);
        return values;
    }

    private static List<String> orderedCardNames(Game game, Iterable<UUID> ids) {
        List<String> values = new ArrayList<String>();
        if (ids != null) {
            for (UUID id : ids) {
                Card card = game.getCard(id);
                values.add(card == null ? "unresolved" : cardKey(game, card));
            }
        }
        return values;
    }

    private static String cardKey(Game game, Card card) {
        return safe(card.getName()) + "|" + safe(card.getClass().getName())
                + "|owner=" + CabtSemanticOrder.playerIndex(game, card.getOwnerId())
                + "|zone=" + (game.getState() == null || game.getState().getZone(card.getId()) == null
                ? "" : game.getState().getZone(card.getId()).name())
                + "|zcc=" + card.getZoneChangeCounter(game);
    }

    private static String stackKey(Game game, StackObject object) {
        List<String> targets = new ArrayList<String>();
        if (object.getStackAbility() != null && object.getStackAbility().getTargets() != null) {
            object.getStackAbility().getTargets().forEach(target -> {
                for (UUID id : target.getTargets()) {
                    targets.add(CabtSemanticOrder.targetKey(game, id));
                }
            });
        }
        return objectKey(game, object)
                + "|controller=" + CabtSemanticOrder.playerIndex(game, object.getControllerId())
                + "|source=" + CabtSemanticOrder.targetKey(game, object.getSourceId())
                + "|targets=" + targets;
    }

    private static String objectKey(Game game, MageObject object) {
        return safe(object.getName()) + "|" + safe(object.getClass().getName())
                + "|semantic=" + CabtSemanticOrder.targetKey(game, object.getId());
    }

    private static String sha256Text(String value) {
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

    private static String safe(Object value) {
        return value == null ? "" : String.valueOf(value);
    }
}
