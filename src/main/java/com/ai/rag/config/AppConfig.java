package com.ai.rag.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Application configuration loaded from {@code app.properties} (bundled in the JAR).
 *
 * <p>Resolution order for every key (highest priority first):
 * <ol>
 *   <li>Environment variable derived from the key: {@code ollama.url} -> {@code OLLAMA_URL},
 *       {@code chunk.size.chars} -> {@code CHUNK_SIZE_CHARS}</li>
 *   <li>Java system property with the same key ({@code -Dollama.url=...})</li>
 *   <li>Value from {@code app.properties}</li>
 *   <li>Hard-coded default</li>
 * </ol>
 */
public final class AppConfig {

    private final Properties props;

    private AppConfig(Properties props) {
        this.props = props;
    }

    /** Loads the bundled configuration file. */
    public static AppConfig load() {
        Properties props = new Properties();
        try (InputStream in = AppConfig.class.getResourceAsStream("/app.properties")) {
            if (in != null) {
                props.load(in);
            }
        } catch (IOException e) {
            // Fall back to defaults below; the app is still runnable without the file.
            System.err.println("[warn] could not read app.properties: " + e.getMessage());
        }
        return new AppConfig(props);
    }

    /** Reads a string value. */
    public String get(String key, String def) {
        String env = System.getenv(toEnvName(key));
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        String sys = System.getProperty(key);
        if (sys != null && !sys.isBlank()) {
            return sys.trim();
        }
        String val = props.getProperty(key);
        return (val != null && !val.isBlank()) ? val.trim() : def;
    }

    /** Reads an integer value. */
    public int getInt(String key, int def) {
        try {
            return Integer.parseInt(get(key, String.valueOf(def)));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /** Reads a double value. */
    public double getDouble(String key, double def) {
        try {
            return Double.parseDouble(get(key, String.valueOf(def)));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /** Converts a property key to its environment variable name, e.g. {@code a.b-c} -> {@code A_B_C}. */
    private static String toEnvName(String key) {
        return key.toUpperCase().replace('.', '_');
    }

    // --- Frequently used typed accessors -------------------------------

    public String ollamaUrl()          { return get("ollama.url", "http://localhost:11434"); }
    public String model()              { return get("ollama.model", "llama3.2:1b"); }
    public String embeddingProvider()  { return get("embedding.provider", "tfidf").toLowerCase(); }
    public String embeddingModel()     { return get("embedding.model", "nomic-embed-text"); }
    public int chunkSize()             { return getInt("chunk.size.chars", 700); }
    public int chunkOverlap()          { return getInt("chunk.overlap.chars", 120); }
    public int topK()                  { return getInt("retrieval.topk", 3); }
    public double minScore()           { return getDouble("retrieval.min.score", 0.05); }
    public double temperature()        { return getDouble("generation.temperature", 0.2); }
    public int numPredict()            { return getInt("generation.num.predict", 512); }
    public int webPort()               { return getInt("web.port", 8080); }
    public int ollamaTimeoutSeconds()  { return getInt("ollama.timeout.seconds", 60); }
}
