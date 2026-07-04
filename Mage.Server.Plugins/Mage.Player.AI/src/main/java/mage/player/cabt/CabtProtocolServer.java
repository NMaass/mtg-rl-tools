package mage.player.cabt;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;
import mage.cards.Card;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Newline-delimited JSON protocol server: the process boundary that gives a
 * Python harness the CABT-style live-game loop. One request object per stdin
 * line, one response object per stdout line; every response carries
 * {@code "ok"} and failures carry a stable {@code "error"} code plus a
 * human-readable {@code "message"}.
 * <p>
 * Commands: {@code ping}, {@code capabilities}, {@code game_start},
 * {@code game_select}, {@code game_finish}, {@code all_card_data},
 * {@code visualize_data}. Unknown or malformed requests fail closed with an
 * error response; they never guess and never touch the running game. Invalid
 * selections return a structured error and leave the pending decision
 * unchanged, so an agent can retry.
 * <p>
 * The class is transport-independent for testability: {@link #handleLine}
 * maps one request line to one response line; {@link #main} wires it to
 * stdin/stdout.
 */
public final class CabtProtocolServer {

    public static final int PROTOCOL_VERSION = 1;

    private static final Gson GSON = new GsonBuilder()
            .disableHtmlEscaping()
            .serializeNulls()
            .create();

    private final MagicCardDataExporter cardDataExporter = new MagicCardDataExporter();
    private CabtGameSession session;

    /**
     * Handles one request line and returns the response line (without the
     * trailing newline).
     */
    public String handleLine(String requestLine) {
        JsonObject request;
        try {
            JsonElement parsed = JsonParser.parseString(requestLine == null ? "" : requestLine);
            if (!parsed.isJsonObject()) {
                return error(null, "MALFORMED_REQUEST", "request must be a JSON object");
            }
            request = parsed.getAsJsonObject();
        } catch (JsonSyntaxException e) {
            return error(null, "MALFORMED_REQUEST", "request is not valid JSON");
        }
        JsonElement id = request.get("id");
        JsonElement command = request.get("command");
        if (command == null || !command.isJsonPrimitive()) {
            return error(id, "MALFORMED_REQUEST", "request has no command field");
        }
        try {
            switch (command.getAsString()) {
                case "ping":
                    return ping(id);
                case "capabilities":
                    return capabilities(id);
                case "game_start":
                    return gameStart(id, request);
                case "game_select":
                    return gameSelect(id, request);
                case "game_finish":
                    return gameFinish(id);
                case "all_card_data":
                    return allCardData(id);
                case "visualize_data":
                    return visualizeData(id);
                default:
                    return error(id, "UNKNOWN_COMMAND",
                            "unknown command: " + command.getAsString());
            }
        } catch (InvalidSelectionException e) {
            return error(id, e.getMessage(), "selection rejected: " + e.getMessage());
        } catch (CabtDeckFactory.UnknownCardException e) {
            return error(id, "UNKNOWN_CARD", e.getMessage());
        } catch (IllegalStateException e) {
            return error(id, codeFromMessage(e.getMessage()), String.valueOf(e.getMessage()));
        } catch (RuntimeException e) {
            return error(id, "INTERNAL",
                    e.getClass().getSimpleName()
                            + (e.getMessage() == null ? "" : ": " + e.getMessage()));
        }
    }

    private String ping(JsonElement id) {
        JsonObject response = okResponse(id);
        response.addProperty("pong", true);
        return GSON.toJson(response);
    }

    private String capabilities(JsonElement id) {
        JsonObject response = okResponse(id);
        response.addProperty("protocolVersion", PROTOCOL_VERSION);
        JsonArray commands = new JsonArray();
        for (String name : new String[]{"ping", "capabilities", "game_start", "game_select",
                "game_finish", "all_card_data", "visualize_data"}) {
            commands.add(name);
        }
        response.add("commands", commands);
        JsonArray selectTypes = new JsonArray();
        for (MagicSelectType type : MagicSelectType.values()) {
            selectTypes.add(type.name());
        }
        response.add("selectTypes", selectTypes);
        return GSON.toJson(response);
    }

    private String gameStart(JsonElement id, JsonObject request) {
        if (session != null && !session.isFinished()) {
            return error(id, "GAME_ALREADY_ACTIVE",
                    "a game is active; send game_finish first");
        }
        JsonElement decksElement = request.get("decks");
        if (decksElement == null || !decksElement.isJsonArray()
                || decksElement.getAsJsonArray().size() != 2) {
            return error(id, "MALFORMED_REQUEST",
                    "game_start needs \"decks\": [deck0, deck1]");
        }
        List<CabtDeckFactory.Entry> deck0;
        List<CabtDeckFactory.Entry> deck1;
        try {
            deck0 = parseDeck(decksElement.getAsJsonArray().get(0));
            deck1 = parseDeck(decksElement.getAsJsonArray().get(1));
        } catch (IllegalArgumentException e) {
            return error(id, "MALFORMED_REQUEST", e.getMessage());
        }
        CabtGameSession.Config config = parseConfig(request.get("options"));
        session = new CabtGameSession(deck0, deck1, config);
        return eventResponse(id, session.start());
    }

    private String gameSelect(JsonElement id, JsonObject request) {
        if (session == null) {
            return error(id, "NO_ACTIVE_GAME", "no game is active; send game_start first");
        }
        JsonElement selectElement = request.get("select");
        if (selectElement == null || !selectElement.isJsonArray()) {
            return error(id, "MALFORMED_REQUEST",
                    "game_select needs \"select\": [option indices]");
        }
        List<Integer> indices = new ArrayList<Integer>();
        for (JsonElement element : selectElement.getAsJsonArray()) {
            if (!element.isJsonPrimitive() || !element.getAsJsonPrimitive().isNumber()
                    || element.getAsDouble() != Math.floor(element.getAsDouble())) {
                return error(id, "MALFORMED_REQUEST", "select indices must be integers");
            }
            indices.add(element.getAsInt());
        }
        return eventResponse(id, session.select(indices));
    }

    private String gameFinish(JsonElement id) {
        if (session != null) {
            session.finish();
            session = null;
        }
        return GSON.toJson(okResponse(id));
    }

    private String allCardData(JsonElement id) {
        if (session == null) {
            return error(id, "NO_ACTIVE_GAME",
                    "all_card_data exports the active game's deck card pool; send game_start first");
        }
        // one entry per distinct card name: the game-scoped card pool, not
        // one entry per physical copy
        Map<String, Card> byName = new LinkedHashMap<String, Card>();
        for (Card card : session.allDeckCards()) {
            if (!byName.containsKey(card.getName())) {
                byName.put(card.getName(), card);
            }
        }
        JsonObject response = okResponse(id);
        List<MagicCardData> cards = new ArrayList<MagicCardData>();
        for (Card card : byName.values()) {
            cards.add(cardDataExporter.export(card));
        }
        response.add("cards", GSON.toJsonTree(cards));
        return GSON.toJson(response);
    }

    private String visualizeData(JsonElement id) {
        if (session == null) {
            return error(id, "NO_ACTIVE_GAME", "no game is active");
        }
        MagicCurrent current = session.snapshotCurrent();
        JsonObject response = okResponse(id);
        response.addProperty("text", renderBoard(current));
        response.add("state", GSON.toJsonTree(current));
        return GSON.toJson(response);
    }

    private String eventResponse(JsonElement id, CabtGameSession.Event event) {
        JsonObject response = okResponse(id);
        switch (event.kind()) {
            case DECISION:
                response.addProperty("finished", false);
                response.addProperty("sequence", event.sequence());
                response.addProperty("player", event.playerName());
                response.addProperty("playerId", event.playerId());
                response.add("observation", GSON.toJsonTree(event.observation()));
                return GSON.toJson(response);
            case GAME_OVER:
                response.addProperty("finished", true);
                JsonObject result = new JsonObject();
                result.addProperty("winner", event.winner());
                result.add("finalState", GSON.toJsonTree(event.finalState()));
                response.add("result", result);
                return GSON.toJson(response);
            case GAME_ERROR:
            default:
                return error(id, "GAME_ERROR", event.errorMessage());
        }
    }

    /**
     * Deck input: a JSON array of {@code {"name": ..., "count": ...}}
     * entries ({@code count} defaults to 1).
     */
    private static List<CabtDeckFactory.Entry> parseDeck(JsonElement deckElement) {
        if (!deckElement.isJsonArray()) {
            throw new IllegalArgumentException(
                    "a deck must be an array of {\"name\", \"count\"} entries");
        }
        List<CabtDeckFactory.Entry> entries = new ArrayList<CabtDeckFactory.Entry>();
        for (JsonElement element : deckElement.getAsJsonArray()) {
            if (!element.isJsonObject() || !element.getAsJsonObject().has("name")) {
                throw new IllegalArgumentException(
                        "each deck entry must be an object with a \"name\"");
            }
            JsonObject entry = element.getAsJsonObject();
            int count = entry.has("count") ? entry.get("count").getAsInt() : 1;
            entries.add(new CabtDeckFactory.Entry(entry.get("name").getAsString(), count));
        }
        if (entries.isEmpty()) {
            throw new IllegalArgumentException("a deck must not be empty");
        }
        return entries;
    }

    private static CabtGameSession.Config parseConfig(JsonElement optionsElement) {
        CabtGameSession.Config config = new CabtGameSession.Config();
        if (optionsElement == null || !optionsElement.isJsonObject()) {
            return config;
        }
        JsonObject options = optionsElement.getAsJsonObject();
        if (options.has("playerNames") && options.get("playerNames").isJsonArray()) {
            JsonArray names = options.get("playerNames").getAsJsonArray();
            config.playerNames(
                    names.size() > 0 ? names.get(0).getAsString() : null,
                    names.size() > 1 ? names.get(1).getAsString() : null);
        }
        if (options.has("seed")) {
            config.seed(options.get("seed").getAsLong());
        }
        if (options.has("maxTurns")) {
            config.maxTurns(options.get("maxTurns").getAsInt());
        }
        if (options.has("decisionTimeoutSeconds")) {
            config.decisionTimeoutSeconds(options.get("decisionTimeoutSeconds").getAsLong());
        }
        return config;
    }

    /** Plain-text board render, the visualize_data() debugging surface. */
    static String renderBoard(MagicCurrent current) {
        StringBuilder text = new StringBuilder();
        text.append("turn ").append(current.getTurnNumber())
                .append(" | ").append(String.valueOf(current.getPhase()))
                .append(" / ").append(String.valueOf(current.getStep()));
        if (current.isGameEnded()) {
            text.append(" | game over: ").append(String.valueOf(current.getWinner()));
        }
        text.append('\n');
        for (MagicPlayerView player : current.getPlayers()) {
            text.append(String.valueOf(player.getName()))
                    .append("  life=").append(player.getLife())
                    .append("  hand=").append(player.getHandCount())
                    .append("  library=").append(player.getLibraryCount())
                    .append("  graveyard=").append(player.getGraveyardCount())
                    .append('\n');
            for (MagicPermanentView permanent : current.getBattlefield()) {
                if (!String.valueOf(player.getPlayerId()).equals(permanent.getControllerId())) {
                    continue;
                }
                text.append("  [").append(permanent.isTapped() ? "T" : " ").append("] ")
                        .append(String.valueOf(permanent.getRef().getName()));
                if (permanent.getPower() != null && permanent.getToughness() != null) {
                    text.append(' ').append(permanent.getPower())
                            .append('/').append(permanent.getToughness());
                }
                text.append('\n');
            }
        }
        if (!current.getStack().isEmpty()) {
            text.append("stack:\n");
            for (MagicStackObjectView stackObject : current.getStack()) {
                text.append("  ").append(String.valueOf(stackObject.getRef().getName())).append('\n');
            }
        }
        return text.toString();
    }

    private static JsonObject okResponse(JsonElement id) {
        JsonObject response = new JsonObject();
        if (id != null) {
            response.add("id", id);
        }
        response.addProperty("ok", true);
        return response;
    }

    private static String error(JsonElement id, String code, String message) {
        JsonObject response = new JsonObject();
        if (id != null) {
            response.add("id", id);
        }
        response.addProperty("ok", false);
        response.addProperty("error", code);
        response.addProperty("message", message);
        return GSON.toJson(response);
    }

    private static String codeFromMessage(String message) {
        if (message != null) {
            int colon = message.indexOf(':');
            String head = colon >= 0 ? message.substring(0, colon) : message;
            if (head.matches("[A-Z_]+")) {
                return head;
            }
        }
        return "INTERNAL";
    }

    /**
     * stdin/stdout loop: one request line in, one response line out, until
     * EOF. Diagnostics go to stderr; stdout carries protocol lines only.
     */
    public static void main(String[] args) throws IOException {
        CabtProtocolServer server = new CabtProtocolServer();
        BufferedReader in = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintStream out = new PrintStream(System.out, true, "UTF-8");
        System.err.println("cabt-protocol-server ready (protocol version "
                + PROTOCOL_VERSION + ")");
        String line;
        while ((line = in.readLine()) != null) {
            if (line.trim().isEmpty()) {
                continue;
            }
            out.println(server.handleLine(line));
        }
        // EOF: the client is gone; tear the game down so the JVM exits
        server.handleLine("{\"command\": \"game_finish\"}");
    }
}
