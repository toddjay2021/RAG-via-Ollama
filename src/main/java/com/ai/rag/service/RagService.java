package com.ai.rag.service;

import com.ai.rag.config.RagProperties;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;

/**
 * Serves RAG conversations over Server-Sent Events.
 *
 * <p>Event protocol of the {@code /api/chat/stream} endpoint:
 * <ul>
 *   <li>{@code sources} - JSON list of retrieved chunks (file, chunk, score, snippet)</li>
 *   <li>{@code token}    - one JSON-wrapped text delta of the generated answer</li>
 *   <li>{@code done}     - JSON timings once the answer is complete</li>
 *   <li>{@code error}    - JSON message when something failed</li>
 * </ul>
 *
 * <p>Token deltas are wrapped in a small JSON record (never sent as raw
 * strings) because SSE data lines cannot contain unescaped newlines.
 */
@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    /** One retrieved knowledge-base chunk as shown in the UI. */
    public record SourceInfo(String file, int chunk, double score, String snippet) {
    }

    /** Emitted with the final event of a chat stream. */
    public record Timings(long retrievalMs, long generationMs) {
    }

    /** JSON wrapper around a generated text delta. */
    public record TokenPayload(String text) {
    }

    /** JSON wrapper around an error message. */
    public record ErrorPayload(String message) {
    }

    private final RagAssistant assistant;
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;
    private final RagProperties props;

    public RagService(RagAssistant assistant, EmbeddingModel embeddingModel,
                      EmbeddingStore<TextSegment> store, RagProperties props) {
        this.assistant = assistant;
        this.embeddingModel = embeddingModel;
        this.store = store;
        this.props = props;
    }

    // --- Retrieval ------------------------------------------------------------

    /**
     * Retrieves the most relevant chunks for a question — the single source
     * of truth for retrieval rules. Both the UI source list and the prompt
     * context are derived from these matches, so what the user sees is
     * exactly what the model is grounded on.
     */
    public List<EmbeddingMatch<TextSegment>> retrieveMatches(String question) {
        Embedding queryEmbedding = embeddingModel.embed(question).content();
        // A query that shares no vocabulary with the corpus (e.g. small talk
        // against the lexical TF-IDF provider) produces a zero vector; cosine
        // similarity is undefined for it and the store would hand back
        // meaningless mid-range scores for arbitrary chunks. Nothing matches,
        // by definition.
        float[] queryVector = queryEmbedding.vector();
        double sumSquares = 0.0;
        for (float v : queryVector) {
            sumSquares += (double) v * v;
        }
        if (sumSquares == 0.0) {
            return List.of();
        }
        EmbeddingSearchResult<TextSegment> result = store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(props.retrieval().topK())
                .minScore(props.retrieval().minScore())
                .build());

        List<EmbeddingMatch<TextSegment>> matches = result.matches();
        if (matches.isEmpty()) {
            return List.of();
        }
        // Relative threshold: absolute score scales differ wildly between
        // embedding providers (TF-IDF tops out around 0.6, nomic-embed-text
        // floors around 0.7 even for unrelated text), so tail chunks are cut
        // relative to the best hit instead of against a fixed number. The
        // absolute floor (min-score) stays as the "everything is weak" guard.
        double cutoff = matches.get(0).score() - props.retrieval().relativeMargin();
        return matches.stream()
                .filter(match -> match.score() >= cutoff)
                .toList();
    }

    /** The retrieved chunks as shown in the UI (file, chunk, score, snippet). */
    public List<SourceInfo> retrieveSources(String question) {
        return retrieveMatches(question).stream()
                .map(RagService::toSourceInfo)
                .toList();
    }

    /** Formats the retrieved chunks as the context block injected into the prompt. */
    private static String buildContext(List<EmbeddingMatch<TextSegment>> matches) {
        StringBuilder context = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : matches) {
            TextSegment segment = match.embedded();
            String file = segment.metadata().getString("file_name");
            Integer chunk = segment.metadata().getInteger("chunk_index");
            context.append("[source: ").append(file != null ? file : "unknown")
                    .append(" chunk ").append(chunk != null ? chunk : 0).append("]\n")
                    .append(segment.text()).append("\n\n");
        }
        return context.toString();
    }

    private static SourceInfo toSourceInfo(EmbeddingMatch<TextSegment> match) {
        TextSegment segment = match.embedded();
        String file = segment.metadata().getString("file_name");
        Integer chunk = segment.metadata().getInteger("chunk_index");
        String snippet = segment.text().replaceAll("\\s+", " ").strip();
        if (snippet.length() > 160) {
            snippet = snippet.substring(0, 157) + "...";
        }
        return new SourceInfo(
                file != null ? file : "unknown",
                chunk != null ? chunk : 0,
                Math.round(match.score() * 1000.0) / 1000.0,
                snippet);
    }

    // --- Streaming chat ---------------------------------------------------------

    /**
     * Runs one full RAG round and streams it over SSE: first the retrieved
     * sources, then the generated answer token by token, then the timings.
     * Retrieval happens exactly once — the same filtered matches are shown in
     * the UI and injected into the prompt as grounding context.
     */
    public SseEmitter streamChat(String question) {
        SseEmitter emitter = new SseEmitter(0L); // no timeout: generation can take a while

        long t0 = System.nanoTime();
        List<EmbeddingMatch<TextSegment>> matches = retrieveMatches(question);
        List<SourceInfo> sources = matches.stream().map(RagService::toSourceInfo).toList();
        long retrievalMs = (System.nanoTime() - t0) / 1_000_000;

        send(emitter, "sources", sources);
        if (matches.isEmpty()) {
            send(emitter, "token", new TokenPayload(
                    "The knowledge base does not contain anything relevant to this question."));
            send(emitter, "done", new Timings(retrievalMs, 0));
            emitter.complete();
            return emitter;
        }

        long t1 = System.nanoTime();
        TokenStream stream = assistant.chat(question, buildContext(matches));
        stream
                .onNext(token -> send(emitter, "token", new TokenPayload(token)))
                .onComplete(response -> {
                    long generationMs = (System.nanoTime() - t1) / 1_000_000;
                    send(emitter, "done", new Timings(retrievalMs, generationMs));
                    emitter.complete();
                })
                .onError(error -> {
                    log.error("Generation failed", error);
                    send(emitter, "error", new ErrorPayload(String.valueOf(error.getMessage())));
                    emitter.complete();
                })
                .start();
        return emitter;
    }

    private void send(SseEmitter emitter, String event, Object payload) {
        try {
            emitter.send(SseEmitter.event().name(event).data(payload));
        } catch (Exception e) {
            // Browser tab was closed mid-stream; nothing left to deliver.
            log.debug("SSE client disconnected while sending '{}' event: {}", event, e.getMessage());
        }
    }
}
