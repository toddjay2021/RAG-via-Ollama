package com.ai.rag.embedding;

import com.ai.rag.ollama.OllamaClient;

import java.io.IOException;

/**
 * Semantic embeddings computed by an Ollama embedding model via /api/embed.
 *
 * <p>Unlike the built-in TF-IDF provider this captures meaning rather than
 * term overlap, but it needs (a) an Ollama server started with embeddings
 * enabled (e.g. {@code ollama serve --embeddings}) and (b) an embedding model
 * such as {@code nomic-embed-text} pulled locally.
 */
public final class OllamaEmbeddingProvider implements EmbeddingProvider {

    private final OllamaClient client;
    private int dimension = -1;

    public OllamaEmbeddingProvider(OllamaClient client) {
        this.client = client;
    }

    @Override
    public double[] embed(String text) throws IOException, InterruptedException {
        double[] vector = client.embed(text);
        if (dimension < 0) {
            dimension = vector.length;
        }
        return vector;
    }

    @Override
    public int dimension() {
        return Math.max(dimension, 0);
    }

    @Override
    public String describe() {
        return "Ollama embeddings (" + client.embedModel() + ", dim " + Math.max(dimension, 0) + ")";
    }
}
