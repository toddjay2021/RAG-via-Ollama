package com.ai.rag;

import com.ai.rag.config.AppConfig;
import com.ai.rag.ollama.OllamaClient;
import com.ai.rag.rag.DocumentLoader;
import com.ai.rag.rag.RagPipeline;
import com.ai.rag.rag.VectorStore;
import com.ai.rag.web.WebServer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

/**
 * Entry point with three modes:
 *
 * <pre>
 *   java -jar rag-via-ollama.jar                 # interactive CLI (default)
 *   java -jar rag-via-ollama.jar --ask "What is RAG?"
 *   java -jar rag-via-ollama.jar --web [--port 8080]
 * </pre>
 *
 * Common flags: {@code --docs &lt;path&gt;} points both modes at an external
 * knowledge-base folder (.md / .txt) instead of the bundled corpus.
 */
public final class App {

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception e) {
            System.err.println();
            System.err.println("[fatal] " + e.getMessage());
            System.exit(1);
        }
    }

    private static void run(String[] args) throws Exception {
        Argv argv = Argv.parse(args);
        if (argv.help()) {
            printUsage();
            return;
        }
        if (argv.port() != null) {
            System.setProperty("web.port", String.valueOf(argv.port()));
        }
        AppConfig cfg = AppConfig.load();

        printBanner(cfg);

        OllamaClient client = new OllamaClient(
                cfg.ollamaUrl(), cfg.model(), cfg.embeddingModel(), cfg.ollamaTimeoutSeconds());

        // --- Preflight: fail early with actionable messages ------------------
        if (!client.isModelAvailable()) {
            throw new IllegalStateException(
                    "Cannot find model '%s' on the Ollama server at %s.\n"
                            .formatted(cfg.model(), cfg.ollamaUrl())
                            + "  1. Is Ollama running?        -> start it with:  ollama serve\n"
                            + "  2. Is the model downloaded? -> run:            ollama pull " + cfg.model());
        }

        RagPipeline pipeline = new RagPipeline(cfg, client);

        // --- Ingest the knowledge base ---------------------------------------
        List<DocumentLoader.KbDocument> docs = (argv.docsDir() != null)
                ? DocumentLoader.loadFromDirectory(Path.of(argv.docsDir()))
                : DocumentLoader.loadFromClasspath("/docs");
        if (docs.isEmpty()) {
            throw new IllegalStateException("No .md/.txt documents found for the knowledge base"
                    + (argv.docsDir() != null ? " in '" + argv.docsDir() + "'" : " on the classpath"));
        }
        long t0 = System.nanoTime();
        try {
            pipeline.ingest(docs);
        } catch (IOException e) {
            if (cfg.embeddingProvider().equals("ollama")) {
                throw new IllegalStateException(
                        "Ollama embeddings are unavailable: " + e.getMessage()
                        + "\nEither start Ollama with embeddings enabled (ollama serve --embeddings)"
                        + "\nor switch back to the built-in provider with EMBEDDING_PROVIDER=tfidf", e);
            }
            throw e;
        }
        long ingestMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("[kb]   ingested %d files -> %d chunks  (%s, %d ms)%n",
                docs.size(), pipeline.chunkCount(), pipeline.embedding().describe(), ingestMs);

        // --- Dispatch to the selected mode ------------------------------------
        if (argv.web()) {
            new WebServer(pipeline, cfg).start();
        } else if (argv.ask() != null) {
            askOnce(pipeline, argv.ask());
        } else {
            repl(pipeline);
        }
    }

    // --- Modes --------------------------------------------------------------

    /** One-shot question mode: print sources, streamed answer and exit. */
    private static void askOnce(RagPipeline pipeline, String question) throws Exception {
        RagPipeline.QueryResult result = pipeline.answer(question, token -> {
            System.out.print(token);
            System.out.flush();
        });
        System.out.println();
        printSources(result.sources());
        printTiming(result);
    }

    /** Interactive read-eval-print loop. */
    private static void repl(RagPipeline pipeline) throws Exception {
        System.out.println();
        System.out.println("Ask anything about the knowledge base. Type 'exit' to quit.");
        System.out.println();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        while (true) {
            System.out.print("you > ");
            System.out.flush();
            String line = in.readLine();
            if (line == null) break; // EOF (Ctrl+D)
            String question = line.strip();
            if (question.isEmpty()) continue;
            if (question.equalsIgnoreCase("exit") || question.equalsIgnoreCase("quit")) break;

            System.out.println();
            askOnce(pipeline, question);
            System.out.println();
        }
        System.out.println("bye!");
    }

    // --- Pretty printing ------------------------------------------------------

    private static void printSources(List<VectorStore.Hit> hits) {
        if (hits.isEmpty()) {
            System.out.println("sources: (nothing above the similarity threshold)");
            return;
        }
        StringBuilder sb = new StringBuilder("sources:");
        for (VectorStore.Hit hit : hits) {
            sb.append(String.format("%n  - %-34s chunk %-3d score %.3f",
                    hit.source(), hit.chunkIndex() + 1, hit.score()));
        }
        System.out.println(sb);
    }

    private static void printTiming(RagPipeline.QueryResult result) {
        System.out.printf("timing: retrieval %d ms, generation %d ms%n",
                result.retrievalMs(), result.generationMs());
    }

    private static void printBanner(AppConfig cfg) {
        System.out.println();
        System.out.println("===============================================================");
        System.out.println("  RAG via Ollama - local Retrieval-Augmented Generation demo");
        System.out.println("===============================================================");
        System.out.printf("  server   : %s%n", cfg.ollamaUrl());
        System.out.printf("  model    : %s (generation)%n", cfg.model());
        System.out.printf("  embedding: %s%n", cfg.embeddingProvider());
        System.out.println("===============================================================");
    }

    private static void printUsage() {
        System.out.println("""
                Usage:
                  java -jar rag-via-ollama.jar [options]

                Options:
                  --ask <question>   answer one question, then exit
                  --web              start the web UI (default port 8080)
                  --port <n>         web UI port (implies --web)
                  --docs <path>      use an external knowledge-base folder (.md/.txt)
                  --help             show this help

                Configuration is read from app.properties and can be overridden
                with environment variables, e.g. OLLAMA_URL, OLLAMA_MODEL, EMBEDDING_PROVIDER.""");
    }

    // --- Tiny argument parser ---------------------------------------------------

    /** Parsed command-line options (see {@link #printUsage()}). */
    private record Argv(boolean web, Integer port, String ask, String docsDir, boolean help) {

        static Argv parse(String[] args) {
            boolean web = false, help = false;
            Integer port = null;
            String ask = null, docsDir = null;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--web" -> web = true;
                    case "--ask" -> ask = value(args, ++i, "--ask");
                    case "--port" -> {
                        String v = value(args, ++i, "--port");
                        try {
                            port = Integer.parseInt(v);
                            web = true;
                        } catch (NumberFormatException e) {
                            throw new IllegalArgumentException("invalid --port value: " + v);
                        }
                    }
                    case "--docs" -> docsDir = value(args, ++i, "--docs");
                    case "--help", "-h" -> help = true;
                    default -> throw new IllegalArgumentException("unknown option: " + args[i]);
                }
            }
            return new Argv(web, port, ask, docsDir, help);
        }

        private static String value(String[] args, int i, String flag) {
            if (i >= args.length) throw new IllegalArgumentException("missing value for " + flag);
            return args[i];
        }
    }
}
