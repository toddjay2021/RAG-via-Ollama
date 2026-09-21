package com.ai.rag.rag;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * In-memory vector store with brute-force cosine-similarity search.
 *
 * <p>Deliberately simple - a real deployment would swap this for Qdrant,
 * pgvector, Chroma, etc., but the interface ({@link #add} / {@link #search})
 * is the same. For a demo-scale knowledge base of a few hundred chunks a
 * linear scan answers in well under a millisecond.
 */
public final class VectorStore {

    /** One indexed knowledge-base chunk with its embedding. */
    public record Entry(String source, int chunkIndex, String text, double[] vector) {
    }

    /** A retrieval hit. */
    public record Hit(Entry entry, double score) {
        public String source()  { return entry.source(); }
        public int chunkIndex() { return entry.chunkIndex(); }
        public String snippet() {
            String s = entry.text().replaceAll("\\s+", " ").strip();
            return s.length() <= 160 ? s : s.substring(0, 157) + "...";
        }
    }

    private final List<Entry> entries = new ArrayList<>();

    /** Adds an embedded chunk. */
    public synchronized void add(Entry entry) {
        entries.add(entry);
    }

    /** Number of stored chunks. */
    public synchronized int size() {
        return entries.size();
    }

    /**
     * Returns the top-K entries closest to the query vector by cosine
     * similarity, ignoring hits below {@code minScore}.
     */
    public synchronized List<Hit> search(double[] query, int topK, double minScore) {
        List<Hit> hits = new ArrayList<>();
        for (Entry e : entries) {
            double score = cosine(query, e.vector());
            if (score >= minScore) {
                hits.add(new Hit(e, score));
            }
        }
        hits.sort(Comparator.comparingDouble(Hit::score).reversed());
        return hits.size() > topK ? new ArrayList<>(hits.subList(0, topK)) : hits;
    }

    /** Cosine similarity between two vectors of equal length. */
    public static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
        }
        for (double v : a) na += v * v;
        for (double v : b) nb += v * v;
        return (na == 0 || nb == 0) ? 0.0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
