package mage.player.cabt;

import mage.constants.MultiplayerAttackOption;
import mage.constants.PhaseStep;
import mage.constants.RangeOfInfluence;
import mage.game.GameImpl;
import mage.game.match.MatchType;
import mage.game.mulligan.MulliganType;
import mage.game.turn.TurnMod;

import java.util.UUID;

/**
 * A real two-player duel on the actual engine loop for protocol-driven games —
 * a minimal mirror of Mage.Game.TwoPlayerDuel (that plugin is not on this
 * module's classpath). Nothing is stubbed: GameImpl runs turns, priority, the
 * stack, combat, and state-based actions exactly as the server does. The
 * test-side twin is CabtSmokeDuel; this one lives in main because
 * {@link CabtGameSession} builds live games outside the test tree.
 */
final class CabtLiveDuel extends GameImpl {

    CabtLiveDuel() {
        super(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL,
                MulliganType.GAME_DEFAULT.getMulligan(0), 0, 20, 7);
    }

    private CabtLiveDuel(final CabtLiveDuel game) {
        super(game);
    }

    @Override
    public MatchType getGameType() {
        return new CabtLiveDuelType();
    }

    @Override
    public int getNumPlayers() {
        return 2;
    }

    @Override
    protected void init(UUID choosingPlayerId) {
        super.init(choosingPlayerId);
        // same first-turn draw skip as TwoPlayerDuel
        state.getTurnMods().add(new TurnMod(startingPlayerId).withSkipStep(PhaseStep.DRAW));
    }

    @Override
    public CabtLiveDuel copy() {
        return new CabtLiveDuel(this);
    }

    static final class CabtLiveDuelType extends MatchType {

        CabtLiveDuelType() {
            this.name = "CABT Live Duel";
            this.maxPlayers = 2;
            this.minPlayers = 2;
            this.numTeams = 0;
            this.useAttackOption = false;
            this.useRange = false;
            this.sideboardingAllowed = false;
        }

        private CabtLiveDuelType(final CabtLiveDuelType matchType) {
            super(matchType);
        }

        @Override
        public CabtLiveDuelType copy() {
            return new CabtLiveDuelType(this);
        }
    }
}
