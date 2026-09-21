# How this demo is built

This project is a self-contained Retrieval-Augmented Generation application
built on Spring Boot 3 and langchain4j 0.34, talking to a local Ollama
server. It serves a single-page web UI whose answers stream token by token
over Server-Sent Events (SSE).

## Components

- **OllamaStreamingChatModel** - langchain4j's streaming chat model for the
  local Ollama server, configured with llama3.2:1b.
- **RagAssistant** - a one-method interface implemented at runtime by
  langchain4j AiServices. Every call is automatically augmented: the question
  retrieves relevant chunks, they are injected into the prompt together with
  the guardrail system message, and the answer is streamed back.
- **TfIdfEmbeddingModel** - a custom langchain4j EmbeddingModel implementing
  a lexical TF-IDF vector space over the ingested corpus. It needs no extra
  model download, which is why the whole demo runs with a single
  `ollama pull llama3.2:1b`. It can be swapped for an Ollama embedding model
  (nomic-embed-text) by setting `rag.embedding.provider=ollama`.
- **InMemoryEmbeddingStore** - langchain4j's built-in vector store; each
  stored entry is a TextSegment with file name and chunk index metadata.
- **EmbeddingStoreContentRetriever / RetrievalAugmentor** - the retrieval
  leg of the pipeline, wired as Spring beans and shared by the assistant and
  the sources endpoint.
- **KnowledgeBaseService** - startup ingestion: loads .md/.txt documents
  from an external folder or the bundled classpath folder, splits them with
  DocumentSplitters.recursive (700 characters, 120 overlap), fits the TF-IDF
  vocabulary, embeds every segment and fills the embedding store.
- **ChatController / RagService** - REST + SSE endpoints. The chat endpoint
  sends a `sources` event first (files, scores, snippets), then `token`
  events as the model generates, then a `done` event with timings.

## Answering a question, step by step

1. The browser opens an EventSource to `/api/chat/stream?question=...`.
2. The question is embedded with the same model used at ingestion time.
3. The embedding store ranks every segment by cosine similarity and returns
   the top 3. Two thresholds then apply: chunks under the absolute minimum
   score are dropped (guarding against "everything is weak"), and chunks
   scoring far below the best hit are dropped by a relative margin, so a
   strong match is not diluted by noise. The survivors are sent to the
   browser first.
4. AiServices builds the augmented prompt: the system guardrails, the
   retrieved context excerpts with their source files, and the question.
5. llama3.2:1b generates the answer and every token is forwarded to the
   browser as an SSE `token` event, followed by a final `done` event.

## Configuration

All knobs live in `application.properties` under the `rag.*` prefix (model
name, embedding provider, chunk size, overlap, top K, minimum score,
temperature, web port) and every value can be overridden with environment
variables such as `RAG_OLLAMA_MODEL` or command-line flags like
`--rag.embedding.provider=ollama`.

## Swapping parts

Each building block is a Spring bean, so production variants are drop-in:
a persistent vector database (pgvector, Qdrant) instead of the in-memory
store, nomic-embed-text instead of TF-IDF, a larger model such as
llama3.2:3b, or per-user chat memory and tools on the same RagAssistant.
