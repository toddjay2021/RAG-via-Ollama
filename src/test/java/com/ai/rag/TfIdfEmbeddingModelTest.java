package com.ai.rag;

import com.ai.rag.embedding.TfIdfEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fast, dependency-free checks that the custom TF-IDF {@code EmbeddingModel}
 * produces sane vectors and that retrieval through a langchain4j embedding
 * store ranks a related chunk above an unrelated one.
 */
class TfIdfEmbeddingModelTest {

    private static final List<String> CORPUS = List.of(
            "chunking splits documents into overlapping pieces for retrieval",
            "ollama serves local language models through a rest api",
            "cosine similarity ranks vectors by the angle between them",
            "the embedding store keeps vectors and answers top-k queries",
            "chunking strategy affects retrieval quality significantly");

    @Test
    void ranksRelatedChunkFirstThroughEmbeddingStore() {
        TfIdfEmbeddingModel model = new TfIdfEmbeddingModel();
        List<TextSegment> segments = CORPUS.stream().map(TextSegment::from).toList();
        model.fit(segments);

        InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(model.embedAll(segments).content(), segments);

        Embedding query = model.embed("how does document chunking work").content();
        EmbeddingSearchResult<TextSegment> result = store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(query)
                .maxResults(3)
                .minScore(0.0)
                .build());

        List<EmbeddingMatch<TextSegment>> matches = result.matches();
        assertEquals(3, matches.size());
        // Lexical TF-IDF cannot split "document" from "documents", so both
        // chunking-related chunks tie on the shared term; what matters is that
        // they rank above every unrelated chunk.
        List<String> topTwo = List.of(matches.get(0).embedded().text(), matches.get(1).embedded().text());
        assertTrue(topTwo.contains(CORPUS.get(0)));
        assertTrue(topTwo.contains(CORPUS.get(4)));
        assertTrue(matches.get(0).score() > 0.0);
    }

    @Test
    void embedsQueriesInSameVectorSpaceAsCorpus() {
        TfIdfEmbeddingModel model = new TfIdfEmbeddingModel();
        List<TextSegment> segments = CORPUS.stream().map(TextSegment::from).toList();
        model.fit(segments);

        Embedding corpus = model.embed(CORPUS.get(0)).content();
        Embedding query = model.embed("chunking splits documents into overlapping pieces for retrieval").content();

        assertEquals(model.dimension(), corpus.vector().length);
        assertEquals(model.dimension(), query.vector().length);
        // Identical texts embed to identical unit-length vectors.
        assertEquals(1.0, cosine(corpus.vector(), query.vector()), 1e-6);
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
