package com.ai.rag.rag;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Splits a document into overlapping chunks.
 *
 * <p>Strategy (a classic, explainable middle ground between naive fixed
 * windows and structure-aware splitting):
 * <ol>
 *   <li>Normalize line endings and split the text into paragraphs.</li>
 *   <li>Split very long paragraphs further into sentences.</li>
 *   <li>Greedily pack consecutive units into chunks of at most
 *       {@code maxChars}, starting every new chunk with a small tail
 *       ({@code overlapChars}) of the previous one so context is preserved
 *       across boundaries.</li>
 * </ol>
 */
public final class DocumentChunker {

    private static final Pattern PARAGRAPH_SPLIT = Pattern.compile("\\n\\s*\\n");
    private static final Pattern SENTENCE_SPLIT = Pattern.compile("(?<=[.!?])\\s+");

    private final int maxChars;
    private final int overlapChars;

    public DocumentChunker(int maxChars, int overlapChars) {
        if (maxChars <= 0 || overlapChars < 0 || overlapChars >= maxChars) {
            throw new IllegalArgumentException("Require 0 <= overlap < size");
        }
        this.maxChars = maxChars;
        this.overlapChars = overlapChars;
    }

    /**
     * Chunks one document.
     *
     * @param text     full document text
     * @param maxChars target maximum chunk length in characters
     * @param overlap  number of trailing characters reused as the head of the next chunk
     */
    public List<String> chunk(String text) {
        String normalized = text.replace("\r\n", "\n").replace('\r', '\n').strip();
        List<String> units = new ArrayList<>();
        for (String paragraph : PARAGRAPH_SPLIT.split(normalized)) {
            String p = paragraph.strip();
            if (p.isEmpty()) continue;
            if (p.length() <= maxChars) {
                units.add(p);
            } else {
                // Paragraph itself is too long: fall back to sentence units.
                for (String sentence : SENTENCE_SPLIT.split(p)) {
                    String s = sentence.strip();
                    if (s.isEmpty()) continue;
                    if (s.length() <= maxChars) {
                        units.add(s);
                    } else {
                        // Single monstrous "sentence": hard windowing.
                        for (int i = 0; i < s.length(); i += maxChars) {
                            units.add(s.substring(i, Math.min(s.length(), i + maxChars)));
                        }
                    }
                }
            }
        }

        // Greedy packing with overlap tails.
        List<String> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String unit : units) {
            if (current.length() > 0 && current.length() + unit.length() + 1 > maxChars) {
                chunks.add(current.toString());
                String tail = current.substring(Math.max(0, current.length() - overlapChars));
                current = new StringBuilder(tail.isEmpty() ? "" : tail + " " + unit);
                if (current.length() > maxChars) {
                    current = new StringBuilder(unit); // degenerate case: tail + unit overflows
                }
            } else {
                if (current.length() > 0) current.append(' ');
                current.append(unit);
            }
        }
        if (current.length() > 0) {
            chunks.add(current.toString());
        }
        return chunks;
    }
}
