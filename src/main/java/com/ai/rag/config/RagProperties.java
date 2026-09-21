package com.ai.rag.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

import java.time.Duration;

/**
 * All demo knobs, bound from {@code application.properties} under the
 * {@code rag.*} prefix. Standard Spring relaxed-binding rules apply, so the
 * same values can be overridden with environment variables such as
 * {@code RAG_OLLAMA_MODEL} or command-line flags like
 * {@code --rag.embedding.provider=ollama}.
 */
@ConfigurationProperties(prefix = "rag")
public record RagProperties(Ollama ollama, Embedding embedding, Chunking chunking,
                            Retrieval retrieval, Generation generation, Docs docs) {

    /** Optional external knowledge-base folder (.md/.txt); empty = bundled classpath docs. */
    public record Docs(String path) {
    }

    public record Ollama(
            String baseUrl,
            String model,
            String embeddingModel,
            @DefaultValue("60s") Duration timeout) {
    }

    /** tfidf (built-in, zero extra downloads) or ollama (semantic, /api/embed). */
    public record Embedding(@DefaultValue("tfidf") String provider) {
    }

    public record Chunking(
            @DefaultValue("700") int size,
            @DefaultValue("120") int overlap) {
    }

    public record Retrieval(
            @DefaultValue("3") int topK,
            @DefaultValue("0.05") double minScore,
            @DefaultValue("0.05") double relativeMargin) {
    }

    public record Generation(
            @DefaultValue("0.2") double temperature,
            @DefaultValue("512") int numPredict) {
    }
}
