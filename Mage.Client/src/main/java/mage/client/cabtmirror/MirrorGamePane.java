package mage.client.cabtmirror;

import mage.client.game.GamePane;

/**
 * The Arena mirror's game tab: a standard {@link GamePane} whose
 * {@code removeGame()} is a no-op. {@code GamePanel.showGame} closes its pane
 * when the server join fails — the mirror has no server on purpose, so the
 * pane must survive that failure. Everything else (layout, panel install,
 * title updates) is stock XMage.
 */
public final class MirrorGamePane extends GamePane {

    @Override
    public void removeGame() {
        // no server to leave: never let a failed session join close the pane
    }
}
