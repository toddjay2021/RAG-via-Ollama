package com.ai.rag.config;

import com.ai.rag.embedding.TfIdfEmbeddingModel;
import com.ai.rag.service.RagAssistant;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
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
 *   EmbeddingModel  ------------+--&gt; EmbeddingStoreContentRetriever ---&gt; RetrievalAugmentor ---+
 *   InMemoryEmbeddingStore -----+                                                            |
 *                                                                                             |
 *   RagAssistant uses the augmentor to ground every answer in retrieved context &lt;------------+
 * </pre>
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

    @Bean
    public EmbeddingStoreContentRetriever contentRetriever(InMemoryEmbeddingStore<TextSegment> store,
                                                            EmbeddingModel embeddingModel,
                                                            RagProperties props) {
        return new EmbeddingStoreContentRetriever(
                store, embeddingModel, props.retrieval().topK(), props.retrieval().minScore());
    }

    @Bean
    public RetrievalAugmentor retrievalAugmentor(EmbeddingStoreContentRetriever contentRetriever) {
        return DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .build();
    }

    // --- Assistant -----------------------------------------------------------

    /**
     * The RAG chat interface implemented by langchain4j AiServices: every call
     * is automatically augmented with retrieved context and answered in
     * streaming mode via a {@link dev.langchain4j.service.TokenStream}.
     */
    @Bean
    public RagAssistant ragAssistant(OllamaStreamingChatModel streamingModel,
                                     RetrievalAugmentor retrievalAugmentor) {
        return AiServices.builder(RagAssistant.class)
                .streamingChatLanguageModel(streamingModel)
                .retrievalAugmentor(retrievalAugmentor)
                .systemMessageProvider(memoryId -> SYSTEM_PROMPT)
                .build();
    }
}
