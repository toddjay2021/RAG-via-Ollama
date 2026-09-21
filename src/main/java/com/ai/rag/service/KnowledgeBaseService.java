package com.ai.rag.service;

import com.ai.rag.config.RagProperties;
import com.ai.rag.embedding.TfIdfEmbeddingModel;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;
import org.springframework.util.StreamUtils;
import org.springframework.web.client.RestClient;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Loads the knowledge base at startup and feeds it into the embedding store:
 * load documents, split them into overlapping segments, fit the TF-IDF
 * vocabulary (when that provider is active), embed every segment and add it
 * to the store.
 *
 * <p>Documents are read from an optional external folder ({@code rag.docs.path})
 * or, by default, from the {@code docs/} folder bundled with the JAR.
 */
@Service
public class KnowledgeBaseService {

    private static final Logger log = LoggerFactory.getLogger(KnowledgeBaseService.class);

    private final RagProperties props;
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;
    private final RestClient restClient = RestClient.create();

    private int fileCount = 0;
    private int segmentCount = 0;

    public KnowledgeBaseService(RagProperties props, EmbeddingModel embeddingModel,
                               EmbeddingStore<TextSegment> store) {
        this.props = props;
        this.embeddingModel = embeddingModel;
        this.store = store;
    }

    /** Fails fast with actionable messages, then ingests the corpus. */
    @PostConstruct
    void initialize() {
        checkOllamaServer();
        try {
            ingest();
        } catch (Exception e) {
            if ("ollama".equals(props.embedding().provider())) {
                throw new IllegalStateException("""
                        Ollama embeddings are unavailable: %s
                        Start Ollama with embeddings enabled (ollama serve --embeddings) and pull an \
                        embedding model (ollama pull %s), or switch back to the built-in provider \
                        with --rag.embedding.provider=tfidf"""
                        .formatted(e.getMessage(), props.ollama().embeddingModel()), e);
            }
            throw e;
        }
    }

    // --- Ingestion ----------------------------------------------------------

    private void ingest() {
        List<Document> documents = loadDocuments();
        if (documents.isEmpty()) {
            throw new IllegalStateException("No .md/.txt documents found for the knowledge base");
        }

        var splitter = DocumentSplitters.recursive(props.chunking().size(), props.chunking().overlap());
        List<TextSegment> segments = new ArrayList<>();
        for (Document document : documents) {
            List<TextSegment> parts = splitter.split(document);
            for (int i = 0; i < parts.size(); i++) {
                // Rebuild each segment with explicit metadata so file name and
                // chunk index survive regardless of splitter internals.
                Metadata metadata = document.metadata().copy()
                        .put("file_name", document.metadata().getString("file_name"))
                        .put("chunk_index", i + 1);
                segments.add(TextSegment.from(parts.get(i).text(), metadata));
            }
        }

        if (embeddingModel instanceof TfIdfEmbeddingModel tfIdf) {
            tfIdf.fit(segments);
        }
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        store.addAll(embeddings, segments);

        fileCount = documents.size();
        segmentCount = segments.size();
        log.info("Ingested {} files -> {} segments ({}, {} dimensions)",
                fileCount, segmentCount, embeddingDescription(), embeddingModel.dimension());
    }

    /** Loads .md/.txt documents from an external folder or the bundled classpath folder. */
    private List<Document> loadDocuments() {
        if (props.docs() != null && props.docs().path() != null && !props.docs().path().isBlank()) {
            return loadFromDirectory(Path.of(props.docs().path()));
        }
        return loadFromClasspath();
    }

    private List<Document> loadFromClasspath() {
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        List<Document> documents = new ArrayList<>();
        for (String pattern : new String[]{"classpath*:/docs/*.md", "classpath*:/docs/*.txt"}) {
            try {
                for (Resource resource : resolver.getResources(pattern)) {
                    if (!resource.isReadable()) continue;
                    String text = StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
                    documents.add(Document.from(text, new Metadata()
                            .put("file_name", resource.getFilename())));
                }
            } catch (IOException e) {
                throw new IllegalStateException("Cannot read knowledge base from " + pattern, e);
            }
        }
        documents.sort((a, b) -> String.valueOf(a.metadata().getString("file_name"))
                .compareTo(String.valueOf(b.metadata().getString("file_name"))));
        return documents;
    }

    private List<Document> loadFromDirectory(Path dir) {
        if (!Files.isDirectory(dir)) {
            throw new IllegalStateException("Configured docs folder does not exist: " + dir);
        }
        List<Document> documents = new ArrayList<>();
        try (Stream<Path> files = Files.walk(dir, 1)) {
            files.filter(Files::isRegularFile)
                 .filter(p -> {
                     String name = p.getFileName().toString().toLowerCase();
                     return name.endsWith(".md") || name.endsWith(".txt");
                 })
                 .sorted()
                 .forEach(p -> {
                     try {
                         documents.add(Document.from(
                                 Files.readString(p, StandardCharsets.UTF_8),
                                 new Metadata().put("file_name", dir.relativize(p).toString())));
                     } catch (IOException e) {
                         log.warn("Skipping unreadable file {}: {}", p, e.getMessage());
                     }
                 });
        } catch (IOException e) {
            throw new IllegalStateException("Cannot walk docs folder " + dir, e);
        }
        return documents;
    }

    // --- Server health --------------------------------------------------------

    /** Fails startup when Ollama is unreachable or the model is missing. */
    private void checkOllamaServer() {
        String url = props.ollama().baseUrl() + "/api/tags";
        try {
            JsonNode models = restClient.get().uri(url).retrieve().body(JsonNode.class);
            boolean installed = false;
            String model = props.ollama().model();
            if (models != null) {
                for (JsonNode m : models.path("models")) {
                    String name = m.path("name").asText("");
                    if (name.equals(model) || name.startsWith(model + ":") || model.startsWith(name)) {
                        installed = true;
                        break;
                    }
                }
            }
            if (!installed) {
                throw new IllegalStateException("""
                        Model '%s' was not found on the Ollama server at %s.
                          1. Is Ollama running?        -> start it with:  ollama serve
                          2. Is the model downloaded? -> run:            ollama pull %s"""
                        .formatted(model, props.ollama().baseUrl(), model));
            }
        } catch (IllegalStateException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("""
                    Cannot reach the Ollama server at %s (%s).
                    Start it with:  ollama serve"""
                    .formatted(props.ollama().baseUrl(), e.getMessage()), e);
        }
    }

    /** Cheap reachability probe for the health endpoint (does not fail). */
    public boolean ollamaUp() {
        try {
            restClient.get().uri(props.ollama().baseUrl() + "/api/tags").retrieve().body(String.class);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // --- Introspection ---------------------------------------------------------

    public int fileCount()     { return fileCount; }
    public int segmentCount()  { return segmentCount; }

    public String embeddingDescription() {
        if (embeddingModel instanceof TfIdfEmbeddingModel tfIdf) {
            return tfIdf.describe();
        }
        return "Ollama embeddings (" + props.ollama().embeddingModel() + ")";
    }
}
