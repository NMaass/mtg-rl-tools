package mage.player.cabt;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Protocol boundary test: drives {@link CabtProtocolServer#handleLine} with
 * raw request lines — exactly what a Python client sends over stdin — and
 * asserts on raw response lines. A real engine game runs underneath; the
 * JSON-level greedy policy mirrors {@link CabtEventPolicy} to prove the whole
 * loop works from serialized observations alone (no Java-side objects), the
 * way an external agent must.
 */
@Timeout(120)
class CabtProtocolServerTest {

    private final CabtProtocolServer server = new CabtProtocolServer();

    @AfterEach
    void tearDown() {
        server.handleLine("{\"command\": \"game_finish\"}");
    }

    private static final String GAME_START = "{\"command\": \"game_start\", \"decks\": ["
            + deckJson() + ", " + deckJson() + "], \"options\": {"
            + "\"playerNames\": [\"Alice\", \"Bob\"], \"seed\": 20260704, \"maxTurns\": 4}}";

    private static String deckJson() {
        return "[{\"name\": \"Forest\", \"count\": 24}, {\"name\": \"Grizzly Bears\", \"count\": 36}]";
    }

    private JsonObject handle(String line) {
        return JsonParser.parseString(server.handleLine(line)).getAsJsonObject();
    }

    // --- transport-level behavior ---

    @Test
    void pingAndCapabilitiesAnswerWithoutAGame() {
        JsonObject pong = handle("{\"command\": \"ping\", \"id\": 7}");
        assertThat(pong.get("ok").getAsBoolean()).isTrue();
        assertThat(pong.get("pong").getAsBoolean()).isTrue();
        assertThat(pong.get("id").getAsInt()).isEqualTo(7);

        JsonObject capabilities = handle("{\"command\": \"capabilities\"}");
        assertThat(capabilities.get("ok").getAsBoolean()).isTrue();
        assertThat(capabilities.get("protocolVersion").getAsInt())
                .isEqualTo(CabtProtocolServer.PROTOCOL_VERSION);
        assertThat(capabilities.get("commands").getAsJsonArray().toString())
                .contains("game_start", "game_select", "game_finish", "all_card_data");
    }

    @Test
    void malformedAndUnknownRequestsFailClosed() {
        assertThat(handle("this is not json").get("error").getAsString())
                .isEqualTo("MALFORMED_REQUEST");
        assertThat(handle("[1, 2]").get("error").getAsString())
                .isEqualTo("MALFORMED_REQUEST");
        assertThat(handle("{\"no\": \"command\"}").get("error").getAsString())
                .isEqualTo("MALFORMED_REQUEST");
        assertThat(handle("{\"command\": \"battle_cheat\"}").get("error").getAsString())
                .isEqualTo("UNKNOWN_COMMAND");
        assertThat(handle("{\"command\": \"game_select\", \"select\": [0]}")
                .get("error").getAsString()).isEqualTo("NO_ACTIVE_GAME");
        assertThat(handle("{\"command\": \"all_card_data\"}").get("error").getAsString())
                .isEqualTo("NO_ACTIVE_GAME");
    }

    @Test
    void unknownCardNamesFailGameStartLoudly() {
        JsonObject response = handle("{\"command\": \"game_start\", \"decks\": ["
                + "[{\"name\": \"No Such Card Ever Printed\", \"count\": 60}], "
                + deckJson() + "]}");
        assertThat(response.get("ok").getAsBoolean()).isFalse();
        assertThat(response.get("error").getAsString()).isEqualTo("UNKNOWN_CARD");
        assertThat(response.get("message").getAsString())
                .contains("No Such Card Ever Printed");
    }

    // --- a real game over the protocol ---

    @Test
    void protocolDrivesARealGameFromStartToFinish() {
        JsonObject response = handle(GAME_START);
        assertThat(response.get("ok").getAsBoolean()).isTrue();

        List<String> selectTypes = new ArrayList<String>();
        boolean sawOwnHand = false;
        int guard = 0;
        while (!response.get("finished").getAsBoolean()) {
            assertThat(guard++).as("decision count stays bounded").isLessThan(2000);
            JsonObject observation = response.get("observation").getAsJsonObject();
            JsonObject select = observation.get("select").getAsJsonObject();
            selectTypes.add(select.get("type").getAsString());

            // hidden-information boundary: only the selecting player's hand
            // is dealt out; every other hand stays counts-only
            int selectingIndex = select.get("playerIndex").getAsInt();
            JsonArray players = observation.get("current").getAsJsonObject()
                    .get("players").getAsJsonArray();
            for (int i = 0; i < players.size(); i++) {
                JsonObject player = players.get(i).getAsJsonObject();
                int handCount = player.get("handCount").getAsInt();
                int handShown = player.get("hand").getAsJsonArray().size();
                if (i == selectingIndex) {
                    sawOwnHand |= handShown > 0 && handShown == handCount;
                } else {
                    assertThat(handShown)
                            .as("opponent hand must not leak (handCount=" + handCount + ")")
                            .isZero();
                }
            }

            response = handle("{\"command\": \"game_select\", \"select\": "
                    + chooseFromJson(select) + "}");
            assertThat(response.get("ok").getAsBoolean())
                    .as("select response: " + response)
                    .isTrue();
        }

        assertThat(selectTypes).contains("PRIORITY", "MULLIGAN");
        assertThat(sawOwnHand).as("the selecting player sees its own hand").isTrue();
        JsonObject result = response.get("result").getAsJsonObject();
        assertThat(result.get("finalState").isJsonObject()).isTrue();

        // after the game ended a fresh game_start must be accepted again
        JsonObject restarted = handle(GAME_START);
        assertThat(restarted.get("ok").getAsBoolean()).isTrue();
    }

    @Test
    void invalidSelectionsReturnStructuredErrorsAndTheGameSurvives() {
        JsonObject first = handle(GAME_START);
        assertThat(first.get("ok").getAsBoolean()).isTrue();
        JsonObject select = first.get("observation").getAsJsonObject()
                .get("select").getAsJsonObject();
        int optionCount = select.get("option").getAsJsonArray().size();

        JsonObject outOfRange = handle(
                "{\"command\": \"game_select\", \"select\": [" + (optionCount + 5) + "]}");
        assertThat(outOfRange.get("ok").getAsBoolean()).isFalse();
        assertThat(outOfRange.get("error").getAsString()).isEqualTo("OPTION_INDEX_OUT_OF_RANGE");

        JsonObject duplicate = handle("{\"command\": \"game_select\", \"select\": [0, 0]}");
        assertThat(duplicate.get("ok").getAsBoolean()).isFalse();
        assertThat(duplicate.get("error").getAsString()).isIn(
                "DUPLICATE_SELECTION", "INVALID_SELECTION_COUNT");

        JsonObject malformed = handle("{\"command\": \"game_select\", \"select\": [0.5]}");
        assertThat(malformed.get("error").getAsString()).isEqualTo("MALFORMED_REQUEST");

        JsonObject secondStart = handle(GAME_START);
        assertThat(secondStart.get("ok").getAsBoolean()).isFalse();
        assertThat(secondStart.get("error").getAsString()).isEqualTo("GAME_ALREADY_ACTIVE");

        // the pending decision survived every rejected request
        JsonObject retried = handle("{\"command\": \"game_select\", \"select\": "
                + chooseFromJson(select) + "}");
        assertThat(retried.get("ok").getAsBoolean()).isTrue();
    }

    @Test
    void cardDataAndVisualizationAreServedDuringAGame() {
        assertThat(handle(GAME_START).get("ok").getAsBoolean()).isTrue();

        JsonObject cardData = handle("{\"command\": \"all_card_data\"}");
        assertThat(cardData.get("ok").getAsBoolean()).isTrue();
        String cards = cardData.get("cards").getAsJsonArray().toString();
        assertThat(cards).contains("Forest", "Grizzly Bears");
        // deck pool is deduplicated by name: 60 physical cards, 2 entries
        assertThat(cardData.get("cards").getAsJsonArray().size()).isEqualTo(2);

        JsonObject visualization = handle("{\"command\": \"visualize_data\"}");
        assertThat(visualization.get("ok").getAsBoolean()).isTrue();
        assertThat(visualization.get("text").getAsString()).contains("turn", "life=");
        assertThat(visualization.get("state").isJsonObject()).isTrue();

        assertThat(handle("{\"command\": \"game_finish\"}").get("ok").getAsBoolean()).isTrue();
        assertThat(handle("{\"command\": \"visualize_data\"}").get("error").getAsString())
                .isEqualTo("NO_ACTIVE_GAME");
    }

    /**
     * JSON-level twin of {@link CabtEventPolicy}: same greedy rules, but read
     * from the serialized select block only.
     */
    private static String chooseFromJson(JsonObject select) {
        String type = select.get("type").getAsString();
        JsonArray options = select.get("option").getAsJsonArray();
        if (type.equals("DECLARE_ATTACKERS") || type.equals("DECLARE_BLOCKERS")) {
            return "[]";
        }
        String[] preferred;
        switch (type) {
            case "PRIORITY":
                preferred = new String[]{"PLAY_LAND", "CAST_SPELL", "PASS_PRIORITY"};
                break;
            case "MULLIGAN":
                preferred = new String[]{"PROMPT_KEEP"};
                break;
            case "PAY_MANA":
                preferred = new String[]{"PROMPT_MANA_SOURCE", "PROMPT_MANA_POOL",
                        "PROMPT_CANCEL_PAYMENT"};
                break;
            default:
                preferred = new String[0];
                break;
        }
        for (String want : preferred) {
            for (int i = 0; i < options.size(); i++) {
                if (options.get(i).getAsJsonObject().get("type").getAsString().equals(want)) {
                    return "[" + i + "]";
                }
            }
        }
        // no rule: take the first minCount options
        int minCount = select.get("minCount").getAsInt();
        StringBuilder indices = new StringBuilder("[");
        for (int i = 0; i < minCount; i++) {
            indices.append(i == 0 ? "" : ", ").append(i);
        }
        return indices.append("]").toString();
    }
}
