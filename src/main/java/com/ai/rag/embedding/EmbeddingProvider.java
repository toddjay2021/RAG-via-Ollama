package com.ai.rag.embedding;

import java.io.IOException;
import java.util.List;

/**
 * Abstraction over text-embedding strategies so the retrieval layer can be
 * swapped between the built-in lexical TF-IDF space and Ollama's semantic
 * embedding endpoint without touching the rest of the pipeline.
 */
public interface EmbeddingProvider {

    /**
     * Optional one-time training step, called once with every chunk of the
     * knowledge base before any vector is produced. Providers that do not need
     * a corpus pass (e.g. Ollama embeddings) can ignore it.
     */
    default void fit(List<String> corpus) {
        // no-op by default
    }

    /** Embeds a single piece of text (a chunk or a user query). */
    double[] embed(String text) throws IOException, InterruptedException;

    /** Dimensionality of the produced vectors. */
    int dimension();

    /** Short human-readable name, shown in the UI and logs. */
    String describe();
}
