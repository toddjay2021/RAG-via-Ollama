package com.ai.rag.config;

import com.ai.rag.embedding.TfIdfEmbeddingModel;
import com.ai.rag.service.RagAssistant;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Wires the langchain4j building blocks together as Spring beans:
 *
 * <pre>
 *   OllamaStreamingChatModel ---&gt; RagAssistant (AiServices, streaming)
 *   InMemoryEmbeddingStore ----+--&gt; RagService (search + filtering + context injection)
 *   EmbeddingModel ------------+        |
 *                                      v
 *                        passes retrieved chunks as the {{context}} template
 *                        variable of RagAssistant.chat(question, context)
 * </pre>
 *
 * <p>Retrieval is deliberately NOT delegated to a langchain4j
 * {@code RetrievalAugmentor}: the application owns the search so that the
 * chunks shown in the UI are exactly the chunks injected into the prompt.
 */
@Configuration(proxyBeanMethods = false)
public class LangChain4jConfig {

    /** Guardrail prompt: the model must stay grounded in the retrieved context. */
    public static final String SYSTEM_PROMPT = """
            You are a precise assistant that answers questions strictly from the provided context
            excerpts of a small knowledge base.

            Rules:
            - Prefer information found in the context; quote or reference the source file when you use it.
            - If the context does not contain the answer, say plainly that the knowledge base does not
              cover it, then optionally give a one-sentence general answer marked as background knowledge.
            - Keep answers concise (2-5 sentences) and well structured.""";

    // --- Models ------------------------------------------------------------

    @Bean
    public OllamaStreamingChatModel ollamaStreamingChatModel(RagProperties props) {
        return OllamaStreamingChatModel.builder()
                .baseUrl(props.ollama().baseUrl())
                .modelName(props.ollama().model())
                .temperature(props.generation().temperature())
                .numPredict(props.generation().numPredict())
                .timeout(props.ollama().timeout())
                .build();
    }

    /**
     * Default embedding model: the built-in lexical TF-IDF vector space.
     * Requires no extra model download, so the whole demo runs with a single
     * {@code ollama pull llama3.2:1b}.
     */
    @Bean
    @ConditionalOnProperty(name = "rag.embedding.provider", havingValue = "tfidf", matchIfMissing = true)
    public TfIdfEmbeddingModel tfIdfEmbeddingModel() {
        return new TfIdfEmbeddingModel();
    }

    /**
     * Optional semantic embedding model backed by Ollama /api/embed.
     * Needs an Ollama server started with embeddings enabled and an embedding
     * model pulled (e.g. nomic-embed-text).
     */
    @Bean
    @ConditionalOnProperty(name = "rag.embedding.provider", havingValue = "ollama")
    public OllamaEmbeddingModel ollamaEmbeddingModel(RagProperties props) {
        return OllamaEmbeddingModel.builder()
                .baseUrl(props.ollama().baseUrl())
                .modelName(props.ollama().embeddingModel())
                .timeout(props.ollama().timeout())
                .build();
    }

    // --- Retrieval ----------------------------------------------------------

    @Bean
    public InMemoryEmbeddingStore<TextSegment> embeddingStore() {
        return new InMemoryEmbeddingStore<>();
    }

    // --- Assistant -----------------------------------------------------------

    /**
     * The RAG chat interface implemented by langchain4j AiServices. Retrieval
     * is owned by the application: the caller injects the retrieved chunks as
     * the {@code context} variable of the {@code @UserMessage} template, so no
     * {@code RetrievalAugmentor} is registered here.
     */
    @Bean
    public RagAssistant ragAssistant(OllamaStreamingChatModel streamingModel) {
        return AiServices.builder(RagAssistant.class)
                .streamingChatLanguageModel(streamingModel)
                .systemMessageProvider(memoryId -> SYSTEM_PROMPT)
                .build();
    }
}
