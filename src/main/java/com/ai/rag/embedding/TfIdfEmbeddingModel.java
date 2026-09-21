package com.ai.rag.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * A self-contained TF-IDF vector space exposed as a langchain4j
 * {@link EmbeddingModel}.
 *
 * <p>This is the default embedding provider because it requires <b>no extra
 * model download and no special Ollama flags</b>: after the knowledge base is
 * ingested, every chunk and every query is mapped to a sparse term vector
 * (term frequency x inverse document frequency), and retrieval compares those
 * vectors with cosine similarity.
 *
 * <p>On small corpora a well-tuned lexical method like TF-IDF is a perfectly
 * competitive baseline against tiny LLM embeddings; for larger, semantically
 * diverse corpora swap in an Ollama embedding model (or any other
 * {@code EmbeddingModel} bean) via {@code rag.embedding.provider=ollama}.
 */
public class TfIdfEmbeddingModel implements EmbeddingModel {

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
    private double[] idf = new double[0];

    private int fittedDocuments = 0;

    /**
     * One-time training step: builds the vocabulary and document frequencies
     * from the corpus of knowledge-base segments. Called by the ingestion
     * service before the corpus is embedded.
     */
    public synchronized void fit(List<TextSegment> corpus) {
        Map<String, Integer> docFreq = new HashMap<>();
        for (TextSegment segment : corpus) {
            fittedDocuments++;
            for (String term : new HashSet<>(tokenize(segment.text()))) {
                docFreq.merge(term, 1, Integer::sum);
            }
        }
        vocabulary.clear();
        List<String> terms = new ArrayList<>(docFreq.keySet());
        terms.sort(String::compareTo); // deterministic dimension order
        idf = new double[terms.size()];
        for (int i = 0; i < terms.size(); i++) {
            vocabulary.put(terms.get(i), i);
            // Smoothed idf: stays positive even for terms appearing in every document.
            idf[i] = Math.log((fittedDocuments + 1.0) / (docFreq.get(terms.get(i)) + 1.0)) + 1.0;
        }
    }

    @Override
    public synchronized Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        if (idf.length == 0) {
            // Defensive auto-fit: if nobody called fit() yet, treat this batch
            // as the corpus. In the normal application flow the ingestion
            // service always fits the model on the full corpus first.
            fit(textSegments);
        }
        List<Embedding> embeddings = new ArrayList<>(textSegments.size());
        for (TextSegment segment : textSegments) {
            embeddings.add(Embedding.from(embedToVector(segment.text())));
        }
        return Response.from(embeddings);
    }

    @Override
    public synchronized int dimension() {
        return vocabulary.size();
    }

    /** Short human-readable description, shown in the UI and logs. */
    public synchronized String describe() {
        return "TF-IDF (built-in, " + vocabulary.size() + " terms, " + fittedDocuments + " fitted chunks)";
    }

    /** Maps one text to its (already length-normalized) TF-IDF vector. */
    private float[] embedToVector(String text) {
        float[] vector = new float[vocabulary.size()];
        Map<String, Integer> counts = new HashMap<>();
        for (String token : tokenize(text)) {
            counts.merge(token, 1, Integer::sum);
        }
        double norm = 0.0;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            Integer idx = vocabulary.get(e.getKey());
            if (idx == null) continue; // unknown query terms are skipped
            double tf = 1.0 + Math.log(e.getValue()); // sublinear term frequency
            vector[idx] = (float) (tf * idf[idx]);
            norm += (double) vector[idx] * vector[idx];
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
        return vector;
    }

    /**
     * Lowercases, strips non-alphanumeric characters and drops stopwords.
     * No stemming is applied on purpose: keeping the exact surface form of
     * each term makes retrieval scores easy to explain while being effective
     * enough for a demo-scale corpus.
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
}
