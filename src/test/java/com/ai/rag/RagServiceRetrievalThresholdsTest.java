package com.ai.rag;

import com.ai.rag.config.RagProperties;
import com.ai.rag.embedding.TfIdfEmbeddingModel;
import com.ai.rag.service.RagService;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the two retrieval thresholds of {@link RagService#retrieveSources}:
 * the absolute floor ({@code min-score}) and the relative margin that drops
 * chunks scoring far below the best hit regardless of the provider's score
 * scale.
 */
class RagServiceRetrievalThresholdsTest {

    private static final String CHUNKING_1 = "chunking splits documents into overlapping pieces for retrieval";
    private static final String CHUNKING_2 = "chunking strategy affects retrieval quality significantly";
    private static final String UNRELATED = "ollama serves local language models through a rest api";

    private RagService service(double minScore, double relativeMargin) {
        TfIdfEmbeddingModel model = new TfIdfEmbeddingModel();
        List<TextSegment> segments = List.of(
                TextSegment.from(CHUNKING_1, new Metadata()
                        .put("file_name", "chunking.md").put("chunk_index", 1)),
                TextSegment.from(CHUNKING_2, new Metadata()
                        .put("file_name", "chunking.md").put("chunk_index", 2)),
                TextSegment.from(UNRELATED, new Metadata()
                        .put("file_name", "ollama.md").put("chunk_index", 1)));
        model.fit(segments);
        InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(model.embedAll(segments).content(), segments);

        RagProperties props = new RagProperties(
                new RagProperties.Ollama("http://localhost:11434", "llama3.2:1b",
                        "nomic-embed-text", Duration.ofSeconds(60)),
                new RagProperties.Embedding("tfidf"),
                new RagProperties.Chunking(700, 120),
                new RagProperties.Retrieval(3, minScore, relativeMargin),
                new RagProperties.Generation(0.2, 512),
                new RagProperties.Docs(""));
        // The assistant is never called by retrieveSources, null is fine here.
        return new RagService(null, model, store, props);
    }

    @Test
    void dropsTailChunksFarBelowTheBestHit() {
        List<RagService.SourceInfo> sources = service(0.0, 0.05)
                .retrieveSources("how does document chunking work");

        // Both chunking chunks score close together and stay; the unrelated
        // ollama chunk lands far below the best hit and is cut by the margin.
        assertEquals(2, sources.size());
        assertTrue(sources.stream().allMatch(s -> s.file().equals("chunking.md")));
    }

    @Test
    void keepsAllTopKHitsWhenScoresAreCloseTogether() {
        List<RagService.SourceInfo> sources = service(0.0, 1.0)
                .retrieveSources("how does document chunking work");

        // A margin of 1.0 covers the whole relevance scale: nothing is cut.
        assertEquals(3, sources.size());
    }

    @Test
    void returnsNothingWhenEverythingIsBelowTheAbsoluteFloor() {
        List<RagService.SourceInfo> sources = service(1.0, 0.05)
                .retrieveSources("how does document chunking work");

        // The store itself filters by min-score, so a floor of 1.0 (only
        // reachable by identical texts) yields no matches at all.
        assertTrue(sources.isEmpty());
    }

    @Test
    void returnsNothingWhenQuerySharesNoVocabularyWithTheCorpus() {
        List<RagService.SourceInfo> sources = service(0.0, 0.05)
                .retrieveSources("purple elephant quantum yoga");

        // None of these words exist in the fitted vocabulary, so the query
        // vector is all zeros and cosine similarity is undefined — the
        // service must return no sources instead of arbitrary 0.5 scores.
        assertTrue(sources.isEmpty());
    }
}
