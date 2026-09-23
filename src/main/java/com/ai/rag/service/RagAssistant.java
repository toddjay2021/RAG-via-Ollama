package com.ai.rag.service;

import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

/**
 * The RAG chat interface, implemented at runtime by langchain4j {@code AiServices}.
 *
 * <p>Retrieval is owned by the application, not by the AI service: the caller
 * searches the knowledge base (applying all filtering rules), then hands the
 * retrieved chunks in as the {@code context} template variable. The grounding
 * rules live in the system prompt (see {@code LangChain4jConfig.SYSTEM_PROMPT}),
 * the user message carries only the context and the question.
 */
public interface RagAssistant {

    /**
     * Streams one grounded answer for the given question.
     *
     * @param question the raw user question
     * @param context  the retrieved knowledge-base chunks, already filtered
     */
    @UserMessage("""
            === Knowledge base context ===
            {{context}}

            === Question ===
            {{question}}
            """)
    TokenStream chat(@V("question") String question, @V("context") String context);
}
