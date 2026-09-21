package com.ai.rag.ollama;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Minimal HTTP client for the local Ollama REST API (https://ollama.com).
 *
 * <p>Only three capabilities are needed by this demo:
 * <ul>
 *   <li>{@link #listModels()} - GET /api/tags</li>
 *   <li>{@link #chatStream(List, Consumer)} - POST /api/chat with NDJSON streaming</li>
 *   <li>{@link #embed(String)} - POST /api/embed</li>
 * </ul>
 */
public final class OllamaClient {

    /** A single chat message exchanged with the model. */
    public record ChatMessage(String role, String content) {
        public static ChatMessage system(String content) { return new ChatMessage("system", content); }
        public static ChatMessage user(String content)   { return new ChatMessage("user", content); }
    }

    private final String baseUrl;
    private final String model;
    private final String embeddingModel;
    private final Duration timeout;
    private final Gson gson = new Gson();
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    public OllamaClient(String baseUrl, String model, String embeddingModel, int timeoutSeconds) {
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.model = model;
        this.embeddingModel = embeddingModel;
        this.timeout = Duration.ofSeconds(timeoutSeconds);
    }

    // --- Server / model info -------------------------------------------

    /**
     * Returns true when the server answers and the configured generation model is installed.
     */
    public boolean isModelAvailable() {
        try {
            for (String name : listModels()) {
                if (name.equals(model) || name.startsWith(model + ":") || model.startsWith(name)) {
                    return true;
                }
            }
            return false;
        } catch (ConnectException e) {
            return false;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    /** Lists installed model names; empty when the server is unreachable. */
    public List<String> listModels() throws IOException, InterruptedException {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/tags"))
                .timeout(timeout)
                .GET()
                .build();
        HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
        List<String> names = new ArrayList<>();
        JsonObject body = gson.fromJson(res.body(), JsonObject.class);
        if (body != null && body.has("models")) {
            JsonArray models = body.getAsJsonArray("models");
            for (var el : models) {
                names.add(el.getAsJsonObject().get("name").getAsString());
            }
        }
        return names;
    }

    // --- Generation -----------------------------------------------------

    /**
     * Runs a streamed chat completion, forwarding every content delta to
     * {@code onToken} as soon as it arrives, and returns the full answer text.
     */
    public String chatStream(List<ChatMessage> messages, double temperature, int numPredict,
                             Consumer<String> onToken) throws IOException, InterruptedException {
        JsonObject payload = new JsonObject();
        payload.addProperty("model", model);
        JsonArray jsonMessages = new JsonArray();
        for (ChatMessage m : messages) {
            JsonObject json = new JsonObject();
            json.addProperty("role", m.role());
            json.addProperty("content", m.content());
            jsonMessages.add(json);
        }
        payload.add("messages", jsonMessages);
        payload.addProperty("stream", true);
        JsonObject options = new JsonObject();
        options.addProperty("temperature", temperature);
        options.addProperty("num_predict", numPredict);
        payload.add("options", options);

        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/chat"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(60))
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload), StandardCharsets.UTF_8))
                .build();

        HttpResponse<java.io.InputStream> res = http.send(req, HttpResponse.BodyHandlers.ofInputStream());
        StringBuilder full = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(res.body(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                JsonObject event = gson.fromJson(line, JsonObject.class);
                if (event.has("error")) {
                    throw new IOException("Ollama error: " + event.get("error").getAsString());
                }
                if (event.has("message")) {
                    String delta = event.getAsJsonObject("message").has("content")
                            ? event.getAsJsonObject("message").get("content").getAsString()
                            : "";
                    if (!delta.isEmpty()) {
                        full.append(delta);
                        if (onToken != null) {
                            onToken.accept(delta);
                        }
                    }
                }
            }
        }
        return full.toString();
    }

    // --- Embeddings ------------------------------------------------------

    /**
     * Computes one embedding vector for the given text via POST /api/embed.
     * Throws {@link IOException} with Ollama's message when the server rejects the call,
     * e.g. when it was started without embeddings enabled.
     */
    public double[] embed(String text) throws IOException, InterruptedException {
        JsonObject payload = new JsonObject();
        payload.addProperty("model", embeddingModel);
        payload.addProperty("input", text);

        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/embed"))
                .header("Content-Type", "application/json")
                .timeout(timeout)
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload), StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
        JsonObject body = gson.fromJson(res.body(), JsonObject.class);
        if (body.has("error")) {
            throw new IOException("Ollama error: " + body.get("error").getAsString());
        }
        JsonArray vector = body.getAsJsonArray("embeddings").get(0).getAsJsonArray();
        double[] out = new double[vector.size()];
        for (int i = 0; i < vector.size(); i++) {
            out[i] = vector.get(i).getAsDouble();
        }
        return out;
    }

    /** For a friendlier health check message when the server is down. */
    public String baseUrl() {
        return baseUrl;
    }

    /** The embedding model used by /api/embed calls. */
    public String embedModel() {
        return embeddingModel;
    }
}
