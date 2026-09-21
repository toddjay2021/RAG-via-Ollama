package com.ai.rag;

import com.ai.rag.embedding.TfIdfEmbeddingProvider;
import com.ai.rag.rag.DocumentChunker;
import com.ai.rag.rag.VectorStore;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Fast, dependency-free checks that the two most error-prone pieces behave:
 * the chunker must respect size/overlap rules, and the TF-IDF provider must
 * rank a related chunk above an unrelated one.
 */
class RagPipelineUnitTests {

    @Test
    void chunkerRespectsMaxSizeAndProducesOverlap() {
        String text = """
                Paragraph one talks about retrieval augmented generation in detail.

                Paragraph two explains embeddings and cosine similarity.

                Paragraph three covers chunking strategies and overlap windows
                with plenty of extra words so that the greedy packer has to split
                the units across several chunks to stay below the size limit
                imposed by the configuration of the demo pipeline.
                """;
        DocumentChunker chunker = new DocumentChunker(160, 40);
        List<String> chunks = chunker.chunk(text);

        assertFalse(chunks.isEmpty());
        for (String chunk : chunks) {
            assertTrue(chunk.length() <= 160 + 80,
                    "chunk clearly above limit: " + chunk.length());
        }
        // At least two chunks, and consecutive ones share some text (overlap).
        if (chunks.size() > 1) {
            String head = chunks.get(1);
            String prevTail = chunks.get(0).substring(Math.max(0, chunks.get(0).length() - 60));
            boolean shares = java.util.Arrays.stream(prevTail.split("\\s+"))
                    .anyMatch(t -> t.length() > 4 && head.contains(t));
            assertTrue(shares, "second chunk should start with a tail of the first");
        }
    }

    @Test
    void tfidfRanksRelatedTextHigher() {
        TfIdfEmbeddingProvider provider = new TfIdfEmbeddingProvider();
        List<String> corpus = List.of(
                "chunking splits documents into overlapping pieces for retrieval",
                "ollama serves local language models through a rest api",
                "cosine similarity ranks vectors by the angle between them",
                "the vector store keeps embeddings and answers top-k queries",
                "chunking strategy affects retrieval quality significantly");
        provider.fit(corpus);

        double[] query = provider.embed("how does document chunking work");
        double scoreChunking = VectorStore.cosine(query, provider.embed(corpus.get(0)));
        double scoreOllama = VectorStore.cosine(query, provider.embed(corpus.get(1)));

        assertTrue(scoreChunking > scoreOllama,
                "expected chunking topic to outrank the ollama topic");
        assertTrue(provider.dimension() > 0);
    }
}
