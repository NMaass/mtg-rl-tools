package mage.player.cabt;

import mage.cards.Card;
import mage.constants.Zone;
import mage.game.Game;
import mage.players.Player;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Omniscient semantic verification state for deterministic replay tests.
 *
 * This is deliberately separate from MagicObservation. Agents must consume the
 * perspective-safe observation; this object exists only to prove that replay
 * reconstruction preserves hidden card zones as well as the visible board.
 * Transport UUIDs are never emitted, so independently rebuilt sessions can be
 * compared directly.
 */
public final class CabtDeterminismSnapshot {

    public static final class CardState {
        private final String name;
        private final String objectClass;
        private final String setCode;
        private final String cardNumber;
        private final int zoneChangeCounter;

        CardState(Card card, Game game) {
            this.name = card == null ? null : card.getName();
            this.objectClass = card == null ? null : card.getClass().getSimpleName();
            this.setCode = card == null ? null : card.getExpansionSetCode();
            this.cardNumber = card == null ? null : card.getCardNumber();
            this.zoneChangeCounter = card == null ? -1 : card.getZoneChangeCounter(game);
        }

        public String getName() { return name; }
        public String getObjectClass() { return objectClass; }
        public String getSetCode() { return setCode; }
        public String getCardNumber() { return cardNumber; }
        public int getZoneChangeCounter() { return zoneChangeCounter; }
    }

    public static final class PlayerState {
        private final int playerIndex;
        private final String name;
        private final int life;
        private final boolean inGame;
        private final boolean passed;
        private final List<CardState> library;
        private final List<CardState> hand;
        private final List<CardState> graveyard;
        private final List<CardState> sideboard;

        PlayerState(int playerIndex, Player player, Game game) {
            this.playerIndex = playerIndex;
            this.name = player.getName();
            this.life = player.getLife();
            this.inGame = player.isInGame();
            this.passed = player.isPassed();
            this.library = cards(game, player.getLibrary().getCardList());
            this.hand = cards(game, player.getHand());
            this.graveyard = cards(game, player.getGraveyard());
            this.sideboard = cards(game, player.getSideboard());
        }

        public int getPlayerIndex() { return playerIndex; }
        public String getName() { return name; }
        public int getLife() { return life; }
        public boolean isInGame() { return inGame; }
        public boolean isPassed() { return passed; }
        public List<CardState> getLibrary() { return library; }
        public List<CardState> getHand() { return hand; }
        public List<CardState> getGraveyard() { return graveyard; }
        public List<CardState> getSideboard() { return sideboard; }
    }

    private final List<PlayerState> players;
    private final MagicCurrent current;
    private final MagicSelectView select;
    private final String eventKind;

    private CabtDeterminismSnapshot(List<PlayerState> players, MagicCurrent current,
                                    MagicSelectView select, String eventKind) {
        this.players = Collections.unmodifiableList(players);
        this.current = current;
        this.select = select;
        this.eventKind = eventKind;
    }

    public static CabtDeterminismSnapshot capture(CabtGameSession session) {
        Game game = session.game();
        List<PlayerState> players = new ArrayList<PlayerState>();
        int index = 0;
        for (UUID playerId : game.getPlayerList()) {
            Player player = game.getPlayer(playerId);
            if (player != null) {
                players.add(new PlayerState(index, player, game));
            }
            index++;
        }

        CabtGameSession.Event event = session.currentEvent();
        MagicCurrent current;
        MagicSelectView select = null;
        String eventKind = event == null ? "NONE" : event.kind().name();
        if (event != null && event.kind() == CabtGameSession.Event.Kind.DECISION) {
            current = event.observation().getCurrent();
            select = event.observation().getSelect();
        } else {
            current = session.snapshotCurrent();
        }
        return new CabtDeterminismSnapshot(players, current, select, eventKind);
    }

    private static List<CardState> cards(Game game, Iterable<UUID> ids) {
        List<CardState> result = new ArrayList<CardState>();
        if (ids != null) {
            for (UUID id : ids) {
                result.add(new CardState(game.getCard(id), game));
            }
        }
        return Collections.unmodifiableList(result);
    }

    public List<PlayerState> getPlayers() { return players; }
    public MagicCurrent getCurrent() { return current; }
    public MagicSelectView getSelect() { return select; }
    public String getEventKind() { return eventKind; }
}
