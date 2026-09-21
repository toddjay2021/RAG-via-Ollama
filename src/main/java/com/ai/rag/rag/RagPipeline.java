package com.ai.rag.rag;

import com.ai.rag.config.AppConfig;
import com.ai.rag.embedding.EmbeddingProvider;
import com.ai.rag.embedding.OllamaEmbeddingProvider;
import com.ai.rag.embedding.TfIdfEmbeddingProvider;
import com.ai.rag.ollama.OllamaClient;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * The heart of the demo: wires together chunking, embedding, vector search
 * and streamed generation into a single Retrieval-Augmented Generation flow.
 *
 * <pre>
 *   documents -&gt; chunks -&gt; embeddings -&gt; vector store
 *                                          |
 *   question  -&gt; embedding --&gt; top-K search -&gt; augmented prompt -&gt; LLM (streamed)
 * </pre>
 */
public final class RagPipeline {

    /** Result of one question: the retrieved sources and the generated answer. */
    public record QueryResult(String question, List<VectorStore.Hit> sources, String answer,
                               long retrievalMs, long generationMs) {
    }

    /** Guardrail prompt: the model must stay grounded in the retrieved context. */
    private static final String SYSTEM_PROMPT = """
            You are a precise assistant that answers questions strictly from the provided context
            excerpts of a small knowledge base.

            Rules:
            - Prefer information found in the context; quote or reference the source file when you use it.
            - If the context does not contain the answer, say plainly that the knowledge base does not
              cover it, then optionally give a one-sentence general answer marked as background knowledge.
            - Keep answers concise (2-5 sentences) and well structured.""";

    private final AppConfig cfg;
    private final OllamaClient client;
    private final EmbeddingProvider embedding;
    private final DocumentChunker chunker;
    private final VectorStore store = new VectorStore();

    private int ingestedFiles = 0;

    public RagPipeline(AppConfig cfg, OllamaClient client) {
        this.cfg = cfg;
        this.client = client;
        this.chunker = new DocumentChunker(cfg.chunkSize(), cfg.chunkOverlap());
        this.embedding = createEmbedding(cfg, client);
    }

    private static EmbeddingProvider createEmbedding(AppConfig cfg, OllamaClient client) {
        return switch (cfg.embeddingProvider()) {
            case "tfidf" -> new TfIdfEmbeddingProvider();
            case "ollama" -> new OllamaEmbeddingProvider(client);
            default -> throw new IllegalArgumentException(
                    "Unknown embedding.provider '" + cfg.embeddingProvider() + "' (expected 'tfidf' or 'ollama')");
        };
    }

    // --- Ingestion --------------------------------------------------------

    /**
     * Ingests documents: chunk, fit the embedding provider on the corpus
     * (TF-IDF needs this), embed every chunk and add it to the store.
     */
    public synchronized void ingest(List<DocumentLoader.KbDocument> documents)
            throws IOException, InterruptedException {
        List<String> texts = new ArrayList<>();
        List<String> sources = new ArrayList<>();
        for (DocumentLoader.KbDocument doc : documents) {
            for (String chunk : chunker.chunk(doc.text())) {
                texts.add(chunk);
                sources.add(doc.fileName());
            }
        }
        embedding.fit(texts);
        for (int i = 0; i < texts.size(); i++) {
            double[] vector = embedding.embed(texts.get(i));
            store.add(new VectorStore.Entry(sources.get(i), i, texts.get(i), vector));
        }
        ingestedFiles += documents.size();
    }

    // --- Retrieval & generation -------------------------------------------

    /** Retrieves the top-K most similar chunks for a question (no generation). */
    public synchronized List<VectorStore.Hit> retrieve(String question)
            throws IOException, InterruptedException {
        double[] queryVector = embedding.embed(question);
        return store.search(queryVector, cfg.topK(), cfg.minScore());
    }

    /**
     * Full RAG round: retrieve relevant chunks, augment a prompt with them and
     * stream the model's answer to {@code onToken}.
     */
    public QueryResult answer(String question, Consumer<String> onToken)
            throws IOException, InterruptedException {
        long t0 = System.nanoTime();
        List<VectorStore.Hit> hits = retrieve(question);
        long retrievalMs = (System.nanoTime() - t0) / 1_000_000;

        if (hits.isEmpty()) {
            return new QueryResult(question, List.of(),
                    "The knowledge base does not contain anything relevant to this question.", retrievalMs, 0);
        }

        long t1 = System.nanoTime();
        String answer = generate(question, hits, onToken);
        long generationMs = (System.nanoTime() - t1) / 1_000_000;
        return new QueryResult(question, hits, answer, retrievalMs, generationMs);
    }

    /**
     * Generation half of the pipeline for callers that already performed
     * retrieval (e.g. the web API, which serves sources and answer separately).
     */
    public String generate(String question, List<VectorStore.Hit> hits, Consumer<String> onToken)
            throws IOException, InterruptedException {
        List<OllamaClient.ChatMessage> messages = List.of(
                OllamaClient.ChatMessage.system(SYSTEM_PROMPT),
                OllamaClient.ChatMessage.user(buildUserPrompt(question, hits)));
        return client.chatStream(messages, cfg.temperature(), cfg.numPredict(), onToken);
    }

    /** Builds the augmented user prompt with numbered context excerpts. */
    private static String buildUserPrompt(String question, List<VectorStore.Hit> hits) {
        StringBuilder sb = new StringBuilder("Context excerpts:\n\n");
        for (int i = 0; i < hits.size(); i++) {
            VectorStore.Hit hit = hits.get(i);
            sb.append('[').append(i + 1).append("] (source: ").append(hit.source())
              .append(", chunk ").append(hit.chunkIndex() + 1)
              .append(", similarity ").append(String.format("%.3f", hit.score())).append(")\n")
              .append(hit.entry().text()).append("\n\n");
        }
        sb.append("Question: ").append(question).append('\n');
        return sb.toString();
    }

    // --- Introspection -----------------------------------------------------

    public synchronized int ingestedFiles() { return ingestedFiles; }
    public int chunkCount()                 { return store.size(); }
    public EmbeddingProvider embedding()    { return embedding; }
    public VectorStore store()              { return store; }
}
