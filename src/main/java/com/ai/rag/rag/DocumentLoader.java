package com.ai.rag.rag;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Loads knowledge-base documents ({@code .md} / {@code .txt}) either from the
 * classpath (docs bundled inside the JAR) or from an external folder.
 */
public final class DocumentLoader {

    /** One loaded document: a file name plus its full text. */
    public record KbDocument(String fileName, String text) {
    }

    /** Lists markdown/text files under a classpath directory, works in IDE and inside a fat JAR. */
    public static List<KbDocument> loadFromClasspath(String resourceDir) throws IOException {
        try {
            var url = DocumentLoader.class.getResource(resourceDir);
            if (url == null) return List.of();
            java.net.URI uri = url.toURI();
            if ("jar".equals(uri.getScheme())) {
                try (var fs = FileSystems.newFileSystem(uri, Map.of())) {
                    return loadFromDirectory(fs.getPath(resourceDir));
                }
            }
            return loadFromDirectory(Path.of(uri));
        } catch (java.net.URISyntaxException e) {
            throw new IOException("Invalid classpath location: " + resourceDir, e);
        }
    }

    /** Lists markdown/text files under an external directory. */
    public static List<KbDocument> loadFromDirectory(Path dir) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) return List.of();
        List<KbDocument> docs = new ArrayList<>();
        try (Stream<Path> files = Files.walk(dir, 1)) {
            files.filter(Files::isRegularFile)
                 .filter(DocumentLoader::isSupported)
                 .sorted()
                 .forEach(p -> {
                     try {
                         docs.add(new KbDocument(
                                 dir.relativize(p).toString(),
                                 Files.readString(p, StandardCharsets.UTF_8)));
                     } catch (IOException e) {
                         System.err.println("[warn] skipping unreadable file " + p + ": " + e.getMessage());
                     }
                 });
        }
        return docs;
    }

    private static boolean isSupported(Path p) {
        String name = p.getFileName().toString().toLowerCase();
        return name.endsWith(".md") || name.endsWith(".txt");
    }
}
