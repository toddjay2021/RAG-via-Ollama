package com.ai.rag.service;

import dev.langchain4j.service.TokenStream;

/**
 * The RAG chat interface, implemented at runtime by langchain4j {@code AiServices}.
 *
 * <p>Every call is automatically augmented: the question is used to retrieve
 * relevant chunks from the embedding store, the retrieved contents are injected
 * into the prompt together with the system guardrails, and llama3.2:1b answers
 * in streaming mode.
 */
public interface RagAssistant {

    /** Streams one grounded answer for the given user question. */
    TokenStream chat(String userMessage);
}
