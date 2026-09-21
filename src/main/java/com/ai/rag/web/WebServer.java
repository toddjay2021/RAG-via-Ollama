package com.ai.rag.web;

import com.ai.rag.config.AppConfig;
import com.ai.rag.rag.RagPipeline;
import com.ai.rag.rag.VectorStore;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.Executors;

/**
 * Zero-dependency web UI + JSON API served by the JDK's built-in HTTP server.
 *
 * <p>Endpoints:
 * <ul>
 *   <li>{@code GET  /}             - single-page chat UI</li>
 *   <li>{@code GET  /api/health}   - service/model/corpus status</li>
 *   <li>{@code POST /api/retrieve}  - {"question": "..."} -> retrieved sources with scores</li>
 *   <li>{@code POST /api/answer}    - {"question": "..."} -> streamed plain-text answer</li>
 * </ul>
 *
 * <p>Retrieval and generation are separate endpoints on purpose: it makes the
 * two RAG stages independently testable (curl-able) and lets the UI render
 * sources before the first generated token arrives.
 */
public final class WebServer {

    private final RagPipeline pipeline;
    private final AppConfig cfg;
    private final Gson gson = new Gson();
    private HttpServer server;

    public WebServer(RagPipeline pipeline, AppConfig cfg) {
        this.pipeline = pipeline;
        this.cfg = cfg;
    }

    /** Starts the server and blocks the calling thread forever. */
    public void start() throws IOException {
        int port = cfg.webPort();
        server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", this::handleIndex);
        server.createContext("/api/health", this::handleHealth);
        server.createContext("/api/retrieve", this::handleRetrieve);
        server.createContext("/api/answer", this::handleAnswer);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("[web] UI ready at http://localhost:" + port + " (Ctrl+C to stop)");
    }

    // --- Handlers ----------------------------------------------------------

    private void handleIndex(HttpExchange ex) throws IOException {
        if (!"GET".equals(ex.getRequestMethod())) {
            respondJson(ex, 405, "{\"error\":\"method not allowed\"}");
            return;
        }
        try (var in = WebServer.class.getResourceAsStream("/web/index.html")) {
            if (in == null) {
                respondJson(ex, 500, "{\"error\":\"index.html missing from classpath\"}");
                return;
            }
            byte[] body = in.readAllBytes();
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(body);
            }
        }
    }

    private void handleHealth(HttpExchange ex) throws IOException {
        JsonObject root = new JsonObject();
        root.addProperty("status", "ok");
        JsonObject ollama = new JsonObject();
        ollama.addProperty("url", cfg.ollamaUrl());
        ollama.addProperty("model", cfg.model());
        root.add("ollama", ollama);
        JsonObject kb = new JsonObject();
        kb.addProperty("files", pipeline.ingestedFiles());
        kb.addProperty("chunks", pipeline.chunkCount());
        root.add("knowledgeBase", kb);
        root.addProperty("embedding", pipeline.embedding().describe());
        respondJson(ex, 200, gson.toJson(root));
    }

    private void handleRetrieve(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            respondJson(ex, 405, "{\"error\":\"method not allowed\"}");
            return;
        }
        try {
            String question = readQuestion(ex);
            List<VectorStore.Hit> hits = pipeline.retrieve(question);
            JsonObject root = new JsonObject();
            root.add("sources", gson.toJsonTree(hits.stream().map(h -> {
                JsonObject o = new JsonObject();
                o.addProperty("file", h.source());
                o.addProperty("chunk", h.chunkIndex() + 1);
                o.addProperty("score", Math.round(h.score() * 1000.0) / 1000.0);
                o.addProperty("snippet", h.snippet());
                return o;
            }).toList()));
            root.addProperty("question", question);
            respondJson(ex, 200, gson.toJson(root));
        } catch (IllegalArgumentException e) {
            respondJson(ex, 400, "{\"error\":\"" + e.getMessage() + "\"}");
        } catch (Exception e) {
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    private void handleAnswer(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            respondJson(ex, 405, "{\"error\":\"method not allowed\"}");
            return;
        }
        String question;
        try {
            question = readQuestion(ex);
        } catch (IllegalArgumentException e) {
            respondJson(ex, 400, "{\"error\":\"" + e.getMessage() + "\"}");
            return;
        }
        // Retrieval is done before headers are sent so failures still get a clean 500.
        List<VectorStore.Hit> hits;
        try {
            hits = pipeline.retrieve(question);
        } catch (Exception e) {
            respondJson(ex, 500, "{\"error\":\"retrieval failed: " + escape(e.getMessage()) + "\"}");
            return;
        }

        ex.getResponseHeaders().set("Content-Type", "text/plain; charset=utf-8");
        ex.sendResponseHeaders(200, 0); // 0 => chunked streaming
        try (OutputStream out = ex.getResponseBody()) {
            if (hits.isEmpty()) {
                out.write("The knowledge base does not contain anything relevant to this question."
                        .getBytes(StandardCharsets.UTF_8));
                out.flush();
                return;
            }
            pipeline.generate(question, hits, token -> {
                try {
                    out.write(token.getBytes(StandardCharsets.UTF_8));
                    out.flush();
                } catch (IOException clientGone) {
                    // Browser aborted the request; surface it to abort generation.
                    throw new RuntimeException(clientGone);
                }
            });
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException io) {
                System.err.println("[web] stream aborted: " + io.getMessage());
            } else {
                System.err.println("[web] generation failed: " + e.getMessage());
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("[web] generation failed: " + e.getMessage());
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // --- Helpers ------------------------------------------------------------

    /** Parses {"question": "..."} from the request body, validating it. */
    private String readQuestion(HttpExchange ex) throws IOException {
        String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        JsonObject json = gson.fromJson(body, JsonObject.class);
        if (json == null || json.get("question") == null) {
            throw new IllegalArgumentException("missing required field: question");
        }
        String question = json.get("question").getAsString().strip();
        if (question.isEmpty()) {
            throw new IllegalArgumentException("question must not be empty");
        }
        return question;
    }

    private void respondJson(HttpExchange ex, int status, String json) throws IOException {
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, body.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(body);
        }
    }

    private static String escape(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
