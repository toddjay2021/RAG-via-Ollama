package com.ai.rag.embedding;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * A self-contained TF-IDF vector space built on the ingested corpus.
 *
 * <p>This is the default embedding provider because it requires <b>no extra
 * model download and no special Ollama flags</b>: after the knowledge base is
 * ingested, every chunk and every query is mapped to a sparse term vector
 * (term frequency x inverse document frequency), and retrieval compares those
 * vectors with cosine similarity.
 *
 * <p>On small corpora a well-tuned lexical method like TF-IDF is a perfectly
 * competitive baseline against tiny LLM embeddings; for larger, semantically
 * diverse corpora swap in {@link OllamaEmbeddingProvider} (or any proper
 * embedding model) via {@code embedding.provider=ollama}.
 */
public final class TfIdfEmbeddingProvider implements EmbeddingProvider {

    /** Common English stopwords removed before counting terms. */
    private static final Set<String> STOPWORDS = Set.of(
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
            "at", "by", "for", "with", "about", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "of", "it", "its",
            "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
            "what", "which", "who", "how", "why", "can", "will", "just", "not", "so",
            "than", "too", "very", "as", "also", "there", "their", "them", "his",
            "her", "our", "your", "my", "me", "us", "him");

    /** term -> dimension index. */
    private final Map<String, Integer> vocabulary = new HashMap<>();

    /** inverse document frequency per dimension. */
    private double[] idf;

    private int fittedDocs = 0;

    @Override
    public void fit(List<String> corpus) {
        Map<String, Integer> docFreq = new HashMap<>();
        for (String doc : corpus) {
            fittedDocs++;
            for (String term : new HashSet<>(tokenize(doc))) {
                docFreq.merge(term, 1, Integer::sum);
            }
        }
        List<String> terms = new ArrayList<>(docFreq.keySet());
        terms.sort(String::compareTo); // deterministic dimension order
        idf = new double[terms.size()];
        for (int i = 0; i < terms.size(); i++) {
            vocabulary.put(terms.get(i), i);
            // Smoothed idf: stays positive even for terms appearing in every document.
            idf[i] = Math.log((fittedDocs + 1.0) / (docFreq.get(terms.get(i)) + 1.0)) + 1.0;
        }
    }

    @Override
    public double[] embed(String text) {
        double[] vec = new double[dimension()];
        if (idf == null) {
            throw new IllegalStateException("TF-IDF provider was not fitted on a corpus yet");
        }
        Map<String, Integer> counts = new HashMap<>();
        for (String token : tokenize(text)) {
            counts.merge(token, 1, Integer::sum);
        }
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            Integer idx = vocabulary.get(e.getKey());
            if (idx == null) continue; // unknown query terms are skipped
            double tf = 1.0 + Math.log(e.getValue()); // sublinear term frequency
            vec[idx] = tf * idf[idx];
        }
        return l2Normalize(vec);
    }

    @Override
    public int dimension() {
        return vocabulary.size();
    }

    @Override
    public String describe() {
        return "TF-IDF (built-in, " + dimension() + " terms, " + fittedDocs + " fitted chunks)";
    }

    /**
     * Lowercases, strips non-alphanumeric characters and drops stopwords.
     * No stemming is applied on purpose: keeping the exact surface form of each
     * term makes retrieval scores easy to explain while being effective enough
     * for a demo-scale corpus.
     */
    private static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        for (String raw : text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (raw.length() >= 2 && !STOPWORDS.contains(raw)) {
                tokens.add(raw);
            }
        }
        return tokens;
    }

    /** Scales the vector to unit length so cosine similarity reduces to a dot product. */
    private static double[] l2Normalize(double[] vec) {
        double norm = 0.0;
        for (double v : vec) norm += v * v;
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vec.length; i++) vec[i] /= norm;
        }
        return vec;
    }
}
