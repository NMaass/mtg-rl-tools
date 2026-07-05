package mage.client.cabtmirror;

import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.GameImpl;
import mage.game.match.MatchType;
import mage.game.mulligan.MulliganType;

/**
 * A puppet {@link GameImpl} for mirroring externally-driven games (MTG
 * Arena). The rules engine loop is never started: {@code play()} is never
 * called and every state change comes from {@link MirrorStateApplier}
 * mutating {@code getState()} directly. The game object exists so real
 * {@code GameView}s can be built for the unmodified XMage client renderer.
 */
public final class MirrorGame extends GameImpl {

    public MirrorGame() {
        super(MultiplayerAttackOption.LEFT, RangeOfInfluence.ALL,
                MulliganType.GAME_DEFAULT.getMulligan(0), 0, 20, 7);
    }

    private MirrorGame(final MirrorGame game) {
        super(game);
    }

    @Override
    public MatchType getGameType() {
        return new MirrorGameType();
    }

    @Override
    public int getNumPlayers() {
        return 2;
    }

    @Override
    public MirrorGame copy() {
        return new MirrorGame(this);
    }

    static final class MirrorGameType extends MatchType {

        MirrorGameType() {
            this.name = "Arena Mirror Duel";
            this.maxPlayers = 2;
            this.minPlayers = 2;
            this.numTeams = 0;
            this.useAttackOption = false;
            this.useRange = false;
            this.sideboardingAllowed = false;
        }

        private MirrorGameType(final MirrorGameType matchType) {
            super(matchType);
        }

        @Override
        public MirrorGameType copy() {
            return new MirrorGameType(this);
        }
    }
}
