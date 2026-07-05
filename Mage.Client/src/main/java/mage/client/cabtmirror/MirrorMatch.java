package mage.client.cabtmirror;

import mage.game.match.MatchImpl;
import mage.game.match.MatchOptions;

/**
 * A no-op {@link MatchImpl} so mirror players have a {@code MatchPlayer}:
 * XMage's {@code PlayerView} reads {@code getMatchPlayer().getWins()}, which
 * NPEs for a bare player. The match itself never runs; it exists only to
 * satisfy that view dependency.
 */
final class MirrorMatch extends MatchImpl {

    MirrorMatch() {
        super(new MatchOptions("Arena Mirror", "Arena Mirror Duel", false));
    }

    @Override
    public void startGame() {
        // the mirror never runs the match loop
    }
}
