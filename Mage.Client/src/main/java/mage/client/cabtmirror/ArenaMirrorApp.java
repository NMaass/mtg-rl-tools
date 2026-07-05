package mage.client.cabtmirror;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;
import mage.cards.repository.CardScanner;
import mage.cards.repository.RepositoryUtil;
import mage.client.MageFrame;
import mage.client.game.GamePanel;
import mage.view.GameView;
import org.apache.log4j.Appender;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import java.util.Enumeration;

import javax.swing.JLayeredPane;
import javax.swing.SwingUtilities;
import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * XMage-client display for live MTG Arena mirroring.
 * <p>
 * Launches the real XMage client ({@link MageFrame}) and, instead of
 * connecting to a server, opens a mirror game pane backed by a puppet
 * {@link MirrorGame}. A newline-delimited JSON loop on stdin/stdout — the
 * same protocol style as {@code CabtProtocolServer} — drives the board:
 * <pre>
 *   {"command":"ping"}
 *   {"command":"mirror_start","players":[{"seat":1,"name":...}],"localSeat":1}
 *   {"command":"mirror_state","state":{...tracker snapshot...}}
 *   {"command":"mirror_message","text":"..."}
 *   {"command":"mirror_game_over","result":"..."}
 *   {"command":"mirror_finish"}
 * </pre>
 * Every request gets one {@code {"ok":...}} response line. Game-state
 * mutation and GameView construction happen on the protocol thread; the
 * finished view is handed to the EDT for rendering.
 */
public final class ArenaMirrorApp {

    private static final Logger logger = Logger.getLogger(ArenaMirrorApp.class);
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();
    private static final int STARTUP_TIMEOUT_SECONDS = 300;

    private MirrorGame game;
    private MirrorStateApplier applier;
    private MirrorGamePane pane;
    private UUID gameId;
    private final AtomicInteger messageId = new AtomicInteger(1);

    public static void main(String[] args) throws Exception {
        // stdout carries protocol lines ONLY: claim a handle on the real
        // fd 1 before anything else can, then divert every other output
        // channel to stderr — plain System.out.println AND log4j's console
        // appender (which grabbed the original System.out at config time).
        PrintStream protocolOut = new PrintStream(
                new FileOutputStream(FileDescriptor.out), true, "UTF-8");
        System.setOut(new PrintStream(
                new FileOutputStream(FileDescriptor.err), true, "UTF-8"));
        redirectConsoleLoggingToStderr();

        boolean prewarm = false;
        for (String arg : args) {
            if ("--prewarm".equals(arg)) {
                prewarm = true;
            }
        }
        if (prewarm) {
            // one-time card database build so the first live run is instant
            RepositoryUtil.bootstrapLocalDb();
            CardScanner.scan();
            protocolOut.println("{\"ok\":true,\"prewarmed\":true}");
            return;
        }

        // the real XMage client startup (splash, look&feel, card db, plugins)
        MageFrame.main(new String[]{});
        waitForMageFrame();
        SwingUtilities.invokeLater(ArenaMirrorApp::hideConnectDialog);

        new ArenaMirrorApp().protocolLoop(protocolOut);
        System.exit(0);
    }

    /**
     * Point every log4j {@link ConsoleAppender} at stderr. XMage logs
     * verbosely to the console appender, which captured the real fd 1 at
     * configuration time; without this, its lines corrupt the protocol
     * stream Python reads from stdout.
     */
    @SuppressWarnings("unchecked")
    private static void redirectConsoleLoggingToStderr() {
        try {
            Enumeration<Appender> appenders =
                    LogManager.getRootLogger().getAllAppenders();
            while (appenders.hasMoreElements()) {
                Appender appender = appenders.nextElement();
                if (appender instanceof ConsoleAppender) {
                    ConsoleAppender console = (ConsoleAppender) appender;
                    console.setTarget(ConsoleAppender.SYSTEM_ERR);
                    console.activateOptions();
                }
            }
        } catch (RuntimeException e) {
            // logging is best-effort; never let it stop the mirror
        }
    }

    private static void waitForMageFrame() throws InterruptedException {
        for (int i = 0; i < STARTUP_TIMEOUT_SECONDS * 10; i++) {
            if (MageFrame.getInstance() != null && MageFrame.getInstance().isVisible()) {
                return;
            }
            Thread.sleep(100);
        }
        throw new IllegalStateException("XMage client window did not start in time");
    }

    /** The connect dialog pops on startup; the mirror needs no server. */
    private static void hideConnectDialog() {
        try {
            Field field = MageFrame.class.getDeclaredField("connectDialog");
            field.setAccessible(true);
            Object dialog = field.get(MageFrame.getInstance());
            if (dialog instanceof java.awt.Component) {
                ((java.awt.Component) dialog).setVisible(false);
            }
        } catch (Exception e) {
            logger.warn("mirror: could not hide connect dialog", e);
        }
    }

    // --- protocol ---

    private void protocolLoop(PrintStream out) throws Exception {
        BufferedReader in = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        System.err.println("arena-mirror-app ready");
        String line;
        while ((line = in.readLine()) != null) {
            if (line.trim().isEmpty()) {
                continue;
            }
            out.println(handleLine(line));
        }
        // EOF: controller is gone
        finishOnEdt();
    }

    String handleLine(String requestLine) {
        JsonObject request;
        try {
            JsonElement parsed = JsonParser.parseString(requestLine);
            if (!parsed.isJsonObject()) {
                return error("MALFORMED_REQUEST", "request must be a JSON object");
            }
            request = parsed.getAsJsonObject();
        } catch (JsonSyntaxException e) {
            return error("MALFORMED_REQUEST", "request is not valid JSON");
        }
        JsonElement command = request.get("command");
        if (command == null || !command.isJsonPrimitive()) {
            return error("MALFORMED_REQUEST", "request has no command field");
        }
        try {
            switch (command.getAsString()) {
                case "ping":
                    return ok().toString();
                case "mirror_start":
                    return mirrorStart(request);
                case "mirror_state":
                    return mirrorState(request);
                case "mirror_message":
                    return mirrorMessage(request);
                case "mirror_game_over":
                    return mirrorGameOver(request);
                case "mirror_screenshot":
                    return mirrorScreenshot(request);
                case "mirror_finish":
                    finishOnEdt();
                    return ok().toString();
                default:
                    return error("UNKNOWN_COMMAND",
                            "unknown command: " + command.getAsString());
            }
        } catch (Exception e) {
            logger.error("mirror: command failed", e);
            return error("INTERNAL", e.getClass().getSimpleName()
                    + (e.getMessage() == null ? "" : ": " + e.getMessage()));
        }
    }

    private String mirrorStart(JsonObject request) throws Exception {
        JsonArray players = request.getAsJsonArray("players");
        if (players == null || players.size() < 2) {
            return error("MALFORMED_REQUEST",
                    "mirror_start needs \"players\": [{seat,name}, ...]");
        }
        Integer localSeat = request.has("localSeat") && !request.get("localSeat").isJsonNull()
                ? request.get("localSeat").getAsInt() : null;

        closeCurrentPane();
        game = new MirrorGame();
        applier = new MirrorStateApplier(game);
        applier.startGame(players, localSeat);
        gameId = game.getId();
        UUID perspective = applier.localPlayerId();

        String title = buildTitle(request);
        SwingUtilities.invokeAndWait(() -> {
            pane = new MirrorGamePane();
            MageFrame.getDesktop().add(pane, JLayeredPane.DEFAULT_LAYER);
            pane.setVisible(true);
            pane.showGame(gameId, gameId, gameId, perspective);
            MageFrame.setActive(pane);
        });
        pushView();
        return ok().toString();
    }

    private String mirrorState(JsonObject request) throws Exception {
        if (game == null) {
            return error("NO_ACTIVE_GAME", "send mirror_start first");
        }
        JsonObject state = request.getAsJsonObject("state");
        if (state == null) {
            return error("MALFORMED_REQUEST", "mirror_state needs \"state\"");
        }
        applier.apply(state);
        pushView();
        return ok().toString();
    }

    private String mirrorMessage(JsonObject request) {
        JsonElement text = request.get("text");
        if (game != null && text != null && text.isJsonPrimitive()) {
            System.err.println("[mirror] " + text.getAsString());
        }
        return ok().toString();
    }

    private String mirrorGameOver(JsonObject request) {
        JsonElement result = request.get("result");
        if (result != null && result.isJsonPrimitive()) {
            System.err.println("[mirror] game over: " + result.getAsString());
        }
        return ok().toString();
    }

    /** Render the mirror window to a PNG (proof/screenshot for the user). */
    private String mirrorScreenshot(JsonObject request) throws Exception {
        JsonElement pathElement = request.get("path");
        if (pathElement == null || !pathElement.isJsonPrimitive()) {
            return error("MALFORMED_REQUEST", "mirror_screenshot needs \"path\"");
        }
        String path = pathElement.getAsString();
        final java.awt.image.BufferedImage[] holder =
                new java.awt.image.BufferedImage[1];
        SwingUtilities.invokeAndWait(() -> {
            java.awt.Component frame = MageFrame.getInstance();
            if (frame == null || frame.getWidth() <= 0) {
                return;
            }
            java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(
                    frame.getWidth(), frame.getHeight(),
                    java.awt.image.BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D graphics = image.createGraphics();
            // paint the live Swing hierarchy — captures the actual rendered
            // board without needing the window to be frontmost
            frame.paint(graphics);
            graphics.dispose();
            holder[0] = image;
        });
        if (holder[0] == null) {
            return error("NO_WINDOW", "mirror window is not ready");
        }
        java.io.File file = new java.io.File(path);
        if (file.getParentFile() != null) {
            file.getParentFile().mkdirs();
        }
        javax.imageio.ImageIO.write(holder[0], "png", file);
        JsonObject response = ok();
        response.addProperty("path", file.getAbsolutePath());
        return response.toString();
    }

    // --- view plumbing ---

    private void pushView() throws Exception {
        if (game == null) {
            return;
        }
        // GameView reads the game on this thread; only rendering goes to EDT
        GameView view = new GameView(game.getState(), game,
                applier.localPlayerId(), null);
        int id = messageId.getAndIncrement();
        SwingUtilities.invokeAndWait(() -> {
            GamePanel panel = MageFrame.getGame(gameId);
            if (panel != null) {
                if (id == 1) {
                    panel.init(id, view, true);
                } else {
                    panel.updateGame(id, view);
                }
            }
        });
    }

    private void closeCurrentPane() throws Exception {
        if (pane == null) {
            return;
        }
        MirrorGamePane closing = pane;
        pane = null;
        SwingUtilities.invokeAndWait(() -> {
            try {
                closing.cleanUp();
                MageFrame.getDesktop().remove(closing);
            } catch (RuntimeException e) {
                logger.warn("mirror: pane close failed", e);
            }
        });
        game = null;
        applier = null;
        gameId = null;
        messageId.set(1);
    }

    private void finishOnEdt() throws Exception {
        closeCurrentPane();
    }

    private static String buildTitle(JsonObject request) {
        StringBuilder title = new StringBuilder("Arena Mirror");
        JsonElement matchId = request.get("matchId");
        if (matchId != null && matchId.isJsonPrimitive()) {
            title.append(" - ").append(matchId.getAsString(), 0,
                    Math.min(8, matchId.getAsString().length()));
        }
        JsonElement gameNumber = request.get("gameNumber");
        if (gameNumber != null && gameNumber.isJsonPrimitive()) {
            title.append(" g").append(gameNumber.getAsString());
        }
        return title.toString();
    }

    private static JsonObject ok() {
        JsonObject response = new JsonObject();
        response.addProperty("ok", true);
        return response;
    }

    private static String error(String code, String message) {
        JsonObject response = new JsonObject();
        response.addProperty("ok", false);
        response.addProperty("error", code);
        response.addProperty("message", message);
        return GSON.toJson(response);
    }
}
