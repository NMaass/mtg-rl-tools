package mage.client.cabtmirror;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import mage.MageInt;
import mage.cards.Card;
import mage.cards.repository.CardInfo;
import mage.cards.repository.CardRepository;
import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentCard;
import mage.game.permanent.PermanentToken;
import mage.game.permanent.token.Token;
import mage.game.permanent.token.custom.CreatureToken;
import mage.game.turn.BeginCombatStep;
import mage.game.turn.BeginningPhase;
import mage.game.turn.CleanupStep;
import mage.game.turn.CombatDamageStep;
import mage.game.turn.CombatPhase;
import mage.game.turn.DeclareAttackersStep;
import mage.game.turn.DeclareBlockersStep;
import mage.game.turn.DrawStep;
import mage.game.turn.EndOfCombatStep;
import mage.game.turn.EndPhase;
import mage.game.turn.EndStep;
import mage.game.turn.Phase;
import mage.game.turn.PostCombatMainPhase;
import mage.game.turn.PostCombatMainStep;
import mage.game.turn.PreCombatMainPhase;
import mage.game.turn.PreCombatMainStep;
import mage.game.turn.Step;
import mage.game.turn.UntapStep;
import mage.game.turn.UpkeepStep;
import mage.players.Player;
import mage.players.StubPlayer;
import org.apache.log4j.Logger;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Applies Arena mirror-state snapshots to a {@link MirrorGame} by direct
 * state mutation — full resync per snapshot, so every state is idempotent
 * and replay can seek. Hidden information stays hidden: objects the log
 * owner cannot see arrive without a name and are placed face-down.
 */
final class MirrorStateApplier {

    private static final Logger logger = Logger.getLogger(MirrorStateApplier.class);
    private static final int LIBRARY_POOL_SIZE = 120;
    private static final String PLACEHOLDER_CARD_NAME = "Plains";

    private final MirrorGame game;
    private final Map<Integer, UUID> seatToPlayerId = new LinkedHashMap<>();
    /** Arena instanceId -> XMage card (for zones that hold Cards). */
    private final Map<Integer, Card> cardsByInstance = new HashMap<>();
    /** Arena instanceId -> XMage permanent id currently on the battlefield. */
    private final Map<Integer, UUID> permanentsByInstance = new HashMap<>();
    /** Face-down filler cards per seat (library + opponent hand counts). */
    private final Map<Integer, java.util.List<Card>> fillerPool = new HashMap<>();
    private final Map<String, CardInfo> cardInfoCache = new HashMap<>();
    private Integer localSeat;

    MirrorStateApplier(MirrorGame game) {
        this.game = game;
    }

    UUID playerIdForSeat(Integer seat) {
        return seat == null ? null : seatToPlayerId.get(seat);
    }

    UUID localPlayerId() {
        return playerIdForSeat(localSeat);
    }

    /** Creates the two players and their filler pools. */
    void startGame(JsonArray players, Integer localSeat) {
        this.localSeat = localSeat;
        MirrorMatch match = new MirrorMatch();
        for (JsonElement element : players) {
            JsonObject playerObject = element.getAsJsonObject();
            int seat = playerObject.get("seat").getAsInt();
            String name = optString(playerObject, "name", "Seat " + seat);
            StubPlayer player = new StubPlayer(name, RangeOfInfluence.ALL);
            mage.cards.decks.Deck deck = new mage.cards.decks.Deck();
            game.addPlayer(player, deck);
            player.initLife(20);
            // PlayerView reads getMatchPlayer().getWins(); give it a match
            player.setMatchPlayer(new mage.game.match.MatchPlayer(player, deck, match));
            seatToPlayerId.put(seat, player.getId());

            java.util.List<Card> pool = new java.util.ArrayList<>();
            for (int i = 0; i < LIBRARY_POOL_SIZE; i++) {
                Card filler = createCardByName(PLACEHOLDER_CARD_NAME);
                if (filler != null) {
                    game.loadCards(Collections.singleton(filler), player.getId());
                    pool.add(filler);
                }
            }
            fillerPool.put(seat, pool);
        }
        game.getState().setTurnNum(1);
        setPhaseStep("Phase_Beginning", "Step_Upkeep");
    }

    /** Full resync of the puppet game to one snapshot. */
    void apply(JsonObject state) {
        JsonObject zones = state.getAsJsonObject("zones");
        applyPlayers(state);
        applyTurnInfo(state);
        if (zones != null) {
            applyBattlefield(zones.getAsJsonArray("battlefield"));
            applyHands(zones.getAsJsonObject("hands"), state);
            applyGraveyards(zones.getAsJsonObject("graveyards"));
            applyExile(zones.getAsJsonArray("exile"));
            applyLibraries(zones, state);
        }
    }

    // --- sections ---

    private void applyPlayers(JsonObject state) {
        JsonArray players = state.getAsJsonArray("players");
        if (players == null) {
            return;
        }
        for (JsonElement element : players) {
            JsonObject playerObject = element.getAsJsonObject();
            UUID playerId = playerIdForSeat(optInt(playerObject, "seat"));
            Player player = game.getPlayer(playerId);
            if (player == null) {
                continue;
            }
            Integer life = optInt(playerObject, "life");
            if (life != null && player.getLife() != life) {
                player.setLife(life, game, null);
            }
        }
    }

    private void applyTurnInfo(JsonObject state) {
        Integer turnNumber = optInt(state, "turnNumber");
        if (turnNumber != null && turnNumber > 0) {
            game.getState().setTurnNum(turnNumber);
        }
        UUID activeId = playerIdForSeat(optInt(state, "activeSeat"));
        if (activeId != null) {
            game.getState().setActivePlayerId(activeId);
        }
        UUID priorityId = playerIdForSeat(optInt(state, "prioritySeat"));
        if (priorityId != null) {
            game.getState().setPriorityPlayerId(priorityId);
        }
        setPhaseStep(optString(state, "phase", null), optString(state, "step", null));
    }

    private void applyBattlefield(JsonArray battlefield) {
        if (battlefield == null) {
            return;
        }
        Set<Integer> present = new HashSet<>();
        for (JsonElement element : battlefield) {
            JsonObject objectJson = element.getAsJsonObject();
            Integer instanceId = optInt(objectJson, "instanceId");
            if (instanceId == null) {
                continue;
            }
            present.add(instanceId);
            UUID controllerId = playerIdForSeat(
                    firstNonNull(optInt(objectJson, "controllerSeat"),
                            optInt(objectJson, "ownerSeat")));
            if (controllerId == null) {
                continue;
            }
            Permanent permanent = existingPermanent(instanceId);
            if (permanent == null) {
                permanent = createPermanent(objectJson, controllerId);
                if (permanent == null) {
                    continue;
                }
                game.getBattlefield().addPermanent(permanent);
                game.getState().setZone(permanent.getId(), Zone.BATTLEFIELD);
                permanentsByInstance.put(instanceId, permanent.getId());
            }
            updatePermanent(permanent, objectJson, controllerId);
        }
        for (Map.Entry<Integer, UUID> entry
                : new HashMap<>(permanentsByInstance).entrySet()) {
            if (!present.contains(entry.getKey())) {
                game.getBattlefield().removePermanent(entry.getValue());
                permanentsByInstance.remove(entry.getKey());
            }
        }
    }

    private void applyHands(JsonObject hands, JsonObject state) {
        if (hands == null) {
            return;
        }
        for (Map.Entry<Integer, UUID> seatEntry : seatToPlayerId.entrySet()) {
            Player player = game.getPlayer(seatEntry.getValue());
            if (player == null) {
                continue;
            }
            JsonArray handArray = hands.getAsJsonArray(String.valueOf(seatEntry.getKey()));
            player.getHand().clear();
            if (handArray == null) {
                continue;
            }
            int fillerIndex = 0;
            for (JsonElement element : handArray) {
                JsonObject objectJson = element.getAsJsonObject();
                Card card = cardForObject(objectJson, seatEntry.getValue());
                if (card == null) {
                    card = takeFiller(seatEntry.getKey(), fillerIndex++);
                }
                if (card != null) {
                    player.getHand().add(card);
                    game.getState().setZone(card.getId(), Zone.HAND);
                }
            }
        }
    }

    private void applyGraveyards(JsonObject graveyards) {
        if (graveyards == null) {
            return;
        }
        for (Map.Entry<Integer, UUID> seatEntry : seatToPlayerId.entrySet()) {
            Player player = game.getPlayer(seatEntry.getValue());
            if (player == null) {
                continue;
            }
            JsonArray graveyardArray =
                    graveyards.getAsJsonArray(String.valueOf(seatEntry.getKey()));
            player.getGraveyard().clear();
            if (graveyardArray == null) {
                continue;
            }
            for (JsonElement element : graveyardArray) {
                Card card = cardForObject(element.getAsJsonObject(),
                        seatEntry.getValue());
                if (card != null) {
                    player.getGraveyard().add(card);
                    game.getState().setZone(card.getId(), Zone.GRAVEYARD);
                }
            }
        }
    }

    private void applyExile(JsonArray exile) {
        game.getExile().getPermanentExile().clear();
        if (exile == null) {
            return;
        }
        for (JsonElement element : exile) {
            JsonObject objectJson = element.getAsJsonObject();
            UUID ownerId = playerIdForSeat(
                    firstNonNull(optInt(objectJson, "ownerSeat"),
                            optInt(objectJson, "controllerSeat")));
            Card card = cardForObject(objectJson, ownerId);
            if (card != null) {
                game.getExile().getPermanentExile().add(card);
                game.getState().setZone(card.getId(), Zone.EXILED);
            }
        }
    }

    private void applyLibraries(JsonObject zones, JsonObject state) {
        JsonObject libraries = zones.getAsJsonObject("libraries");
        for (Map.Entry<Integer, UUID> seatEntry : seatToPlayerId.entrySet()) {
            Player player = game.getPlayer(seatEntry.getValue());
            if (player == null) {
                continue;
            }
            Integer count = null;
            if (libraries != null) {
                count = optInt(libraries, String.valueOf(seatEntry.getKey()));
            }
            if (count == null) {
                count = libraryCountFromPlayers(state, seatEntry.getKey());
            }
            if (count == null) {
                continue;
            }
            player.getLibrary().clear();
            java.util.List<Card> pool = fillerPool.get(seatEntry.getKey());
            if (pool == null) {
                continue;
            }
            for (int i = 0; i < Math.min(count, pool.size()); i++) {
                Card filler = pool.get(pool.size() - 1 - i);
                player.getLibrary().putOnTop(filler, game);
                game.getState().setZone(filler.getId(), Zone.LIBRARY);
            }
        }
    }

    // --- object/card construction ---

    private Permanent existingPermanent(Integer instanceId) {
        UUID permanentId = permanentsByInstance.get(instanceId);
        if (permanentId == null) {
            return null;
        }
        Permanent permanent = game.getBattlefield().getPermanent(permanentId);
        if (permanent == null) {
            permanentsByInstance.remove(instanceId);
        }
        return permanent;
    }

    private Permanent createPermanent(JsonObject objectJson, UUID controllerId) {
        String name = optString(objectJson, "name", null);
        boolean isToken = optBool(objectJson, "isToken");
        boolean faceDown = optBool(objectJson, "faceDown");
        if (name != null && !faceDown && isToken) {
            Token token = buildToken(objectJson, name);
            return new PermanentToken(token, controllerId, game);
        }
        Card card;
        if (name != null && !faceDown) {
            card = createCardByName(name);
            if (card == null) {
                // unknown to XMage (alchemy card, unimported set): show the
                // object face-down rather than dropping it from the board
                card = createCardByName(PLACEHOLDER_CARD_NAME);
                faceDown = true;
            }
        } else {
            card = createCardByName(PLACEHOLDER_CARD_NAME);
            faceDown = true;
        }
        if (card == null) {
            return null;
        }
        game.loadCards(Collections.singleton(card), controllerId);
        PermanentCard permanent = new PermanentCard(card, controllerId, game);
        if (faceDown) {
            permanent.setFaceDown(true, game);
        }
        return permanent;
    }

    private Token buildToken(JsonObject objectJson, String name) {
        Integer power = optInt(objectJson, "power");
        Integer toughness = optInt(objectJson, "toughness");
        CreatureToken token = new CreatureToken(
                power == null ? 0 : power,
                toughness == null ? 0 : toughness);
        token.setName(name);
        return token;
    }

    private void updatePermanent(Permanent permanent, JsonObject objectJson,
                                 UUID controllerId) {
        permanent.setTapped(optBool(objectJson, "tapped"));
        Integer power = optInt(objectJson, "power");
        Integer toughness = optInt(objectJson, "toughness");
        try {
            if (power != null) {
                MageInt current = permanent.getPower();
                if (current.getValue() != power) {
                    current.setModifiedBaseValue(power);
                }
            }
            if (toughness != null) {
                MageInt current = permanent.getToughness();
                if (current.getValue() != toughness) {
                    current.setModifiedBaseValue(toughness);
                }
            }
        } catch (RuntimeException e) {
            logger.warn("mirror: cannot set P/T on " + permanent.getName(), e);
        }
    }

    private Card cardForObject(JsonObject objectJson, UUID ownerId) {
        Integer instanceId = optInt(objectJson, "instanceId");
        String name = optString(objectJson, "name", null);
        if (name == null || optBool(objectJson, "faceDown")) {
            return null;
        }
        if (instanceId != null) {
            Card cached = cardsByInstance.get(instanceId);
            if (cached != null && cached.getName().equals(name)) {
                return cached;
            }
        }
        Card card = createCardByName(name);
        if (card == null) {
            return null;
        }
        game.loadCards(Collections.singleton(card),
                ownerId != null ? ownerId : firstPlayerId());
        if (instanceId != null) {
            cardsByInstance.put(instanceId, card);
        }
        return card;
    }

    private Card createCardByName(String name) {
        CardInfo cardInfo = cardInfoCache.get(name);
        if (cardInfo == null) {
            cardInfo = CardRepository.instance.findPreferredCoreExpansionCard(name);
            if (cardInfo == null) {
                // split/adventure names arrive as "A // B"
                int split = name.indexOf(" // ");
                if (split > 0) {
                    cardInfo = CardRepository.instance
                            .findPreferredCoreExpansionCard(name.substring(0, split));
                }
            }
            if (cardInfo == null) {
                logger.warn("mirror: no XMage card for name: " + name);
                return null;
            }
            cardInfoCache.put(name, cardInfo);
        }
        return cardInfo.createCard();
    }

    private Card takeFiller(Integer seat, int index) {
        java.util.List<Card> pool = fillerPool.get(seat);
        if (pool == null || index >= pool.size()) {
            return null;
        }
        return pool.get(index);
    }

    private UUID firstPlayerId() {
        return seatToPlayerId.values().iterator().next();
    }

    private Integer libraryCountFromPlayers(JsonObject state, Integer seat) {
        JsonArray players = state.getAsJsonArray("players");
        if (players == null) {
            return null;
        }
        for (JsonElement element : players) {
            JsonObject playerObject = element.getAsJsonObject();
            if (seat.equals(optInt(playerObject, "seat"))) {
                return optInt(playerObject, "libraryCount");
            }
        }
        return null;
    }

    // --- phase/step mapping ---

    private void setPhaseStep(String arenaPhase, String arenaStep) {
        Phase phase = phaseFor(arenaPhase, arenaStep);
        if (phase == null) {
            return;
        }
        Step step = stepFor(arenaPhase, arenaStep);
        if (step != null) {
            phase.setStep(step);
        }
        game.getState().getTurn().setPhase(phase);
    }

    private Phase phaseFor(String arenaPhase, String arenaStep) {
        if (arenaPhase == null) {
            return null;
        }
        switch (arenaPhase) {
            case "Phase_Beginning":
                return new BeginningPhase();
            case "Phase_Main1":
                return new PreCombatMainPhase();
            case "Phase_Combat":
                return new CombatPhase();
            case "Phase_Main2":
                return new PostCombatMainPhase();
            case "Phase_Ending":
                return new EndPhase();
            default:
                return null;
        }
    }

    private Step stepFor(String arenaPhase, String arenaStep) {
        if (arenaStep == null) {
            if ("Phase_Main1".equals(arenaPhase)) {
                return new PreCombatMainStep();
            }
            if ("Phase_Main2".equals(arenaPhase)) {
                return new PostCombatMainStep();
            }
            return null;
        }
        switch (arenaStep) {
            case "Step_Untap":
                return new UntapStep();
            case "Step_Upkeep":
                return new UpkeepStep();
            case "Step_Draw":
                return new DrawStep();
            case "Step_BeginCombat":
                return new BeginCombatStep();
            case "Step_DeclareAttack":
                return new DeclareAttackersStep();
            case "Step_DeclareBlock":
                return new DeclareBlockersStep();
            case "Step_FirstStrikeDamage":
                return new CombatDamageStep(true);
            case "Step_CombatDamage":
                return new CombatDamageStep(false);
            case "Step_EndCombat":
                return new EndOfCombatStep();
            case "Step_End":
                return new EndStep();
            case "Step_Cleanup":
                return new CleanupStep();
            default:
                return null;
        }
    }

    // --- tiny JSON helpers ---

    private static Integer optInt(JsonObject object, String field) {
        JsonElement element = object.get(field);
        if (element == null || element.isJsonNull() || !element.isJsonPrimitive()) {
            return null;
        }
        try {
            return element.getAsInt();
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static String optString(JsonObject object, String field,
                                    String defaultValue) {
        JsonElement element = object.get(field);
        if (element == null || element.isJsonNull() || !element.isJsonPrimitive()) {
            return defaultValue;
        }
        return element.getAsString();
    }

    private static boolean optBool(JsonObject object, String field) {
        JsonElement element = object.get(field);
        return element != null && element.isJsonPrimitive()
                && !element.isJsonNull() && element.getAsBoolean();
    }

    private static Integer firstNonNull(Integer first, Integer second) {
        return first != null ? first : second;
    }
}
