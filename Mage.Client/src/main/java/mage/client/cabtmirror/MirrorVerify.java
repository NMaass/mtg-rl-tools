package mage.client.cabtmirror;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import mage.cards.repository.RepositoryUtil;
import mage.constants.CardType;
import mage.view.CardView;
import mage.view.CardsView;
import mage.view.GameView;
import mage.view.PermanentView;
import mage.view.PlayerView;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Headless verification of the mirror rendering pipeline: builds the same
 * {@link MirrorGame} the live display uses, applies recorded snapshots, then
 * constructs the real XMage {@link GameView} from the log-owner's
 * perspective and prints a compact board summary as JSON.
 * <p>
 * This is the same GameView the client's GamePanel renders, so asserting on
 * it proves the display shows the recorded board — including that the
 * opponent's hidden cards render face-down — without needing a live GUI.
 * <pre>
 *   MirrorVerify &lt;mirror_states.jsonl&gt; [stateIndex]
 * </pre>
 * With no index, the last state of the first game is used.
 */
public final class MirrorVerify {

    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: MirrorVerify <mirror_states.jsonl> [stateIndex]");
            System.exit(2);
        }
        RepositoryUtil.bootstrapLocalDb();

        List<JsonObject> states = readStates(args[0]);
        if (states.isEmpty()) {
            System.err.println("no states in " + args[0]);
            System.exit(2);
        }
        String firstMatch = optString(states.get(0), "matchId");
        List<JsonObject> firstGame = new ArrayList<>();
        for (JsonObject state : states) {
            if (!equalsOrNull(optString(state, "matchId"), firstMatch)) {
                break;
            }
            firstGame.add(state);
        }
        int index = args.length >= 2 ? Integer.parseInt(args[1]) : firstGame.size() - 1;
        index = Math.max(0, Math.min(index, firstGame.size() - 1));

        JsonObject firstState = firstGame.get(0);
        MirrorGame game = new MirrorGame();
        MirrorStateApplier applier = new MirrorStateApplier(game);
        applier.startGame(playersArray(firstState), optInt(firstState, "localSeat"));
        for (int i = 0; i <= index; i++) {
            applier.apply(firstGame.get(i));
        }

        GameView view = new GameView(game.getState(), game,
                applier.localPlayerId(), null);
        System.out.println(GSON.toJson(summarize(view, index, firstGame.size())));
    }

    private static JsonObject summarize(GameView view, int index, int total) {
        JsonObject summary = new JsonObject();
        summary.addProperty("stateIndex", index);
        summary.addProperty("gameStateCount", total);
        summary.addProperty("turn", view.getTurn());
        summary.addProperty("phase", String.valueOf(view.getPhase()));
        summary.addProperty("step", String.valueOf(view.getStep()));

        JsonArray players = new JsonArray();
        int faceDownBattlefield = 0;
        int faceUpBattlefield = 0;
        for (PlayerView player : view.getPlayers()) {
            JsonObject playerJson = new JsonObject();
            playerJson.addProperty("name", player.getName());
            playerJson.addProperty("life", player.getLife());
            playerJson.addProperty("handCount", player.getHandCount());
            playerJson.addProperty("libraryCount", player.getLibraryCount());

            JsonArray battlefield = new JsonArray();
            for (PermanentView permanent : player.getBattlefield().values()) {
                JsonObject permanentJson = new JsonObject();
                permanentJson.addProperty("name", permanent.getName());
                permanentJson.addProperty("tapped", permanent.isTapped());
                permanentJson.addProperty("faceDown", permanent.isFaceDown());
                permanentJson.addProperty("isCreature",
                        permanent.getCardTypes().contains(CardType.CREATURE));
                battlefield.add(permanentJson);
                if (permanent.isFaceDown()) {
                    faceDownBattlefield++;
                } else {
                    faceUpBattlefield++;
                }
            }
            playerJson.add("battlefield", battlefield);
            players.add(playerJson);
        }
        summary.add("players", players);

        // hand of the perspective player: real cards; opponent hand stays a count
        JsonArray myHand = new JsonArray();
        int handFaceDown = 0;
        CardsView hand = view.getMyHand();
        if (hand != null) {
            for (Map.Entry<UUID, CardView> entry : hand.entrySet()) {
                CardView card = entry.getValue();
                myHand.add(card.getName());
                if (card.isFaceDown()) {
                    handFaceDown++;
                }
            }
        }
        summary.add("myHand", myHand);
        summary.addProperty("myHandFaceDownCount", handFaceDown);
        summary.addProperty("battlefieldFaceUp", faceUpBattlefield);
        summary.addProperty("battlefieldFaceDown", faceDownBattlefield);
        return summary;
    }

    private static List<JsonObject> readStates(String path) throws Exception {
        List<JsonObject> states = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    states.add(JsonParser.parseString(line).getAsJsonObject());
                }
            }
        }
        return states;
    }

    private static JsonArray playersArray(JsonObject state) {
        return state.getAsJsonArray("players");
    }

    private static String optString(JsonObject object, String field) {
        return object.has(field) && !object.get(field).isJsonNull()
                ? object.get(field).getAsString() : null;
    }

    private static Integer optInt(JsonObject object, String field) {
        return object.has(field) && !object.get(field).isJsonNull()
                ? object.get(field).getAsInt() : null;
    }

    private static boolean equalsOrNull(String a, String b) {
        return a == null ? b == null : a.equals(b);
    }
}
