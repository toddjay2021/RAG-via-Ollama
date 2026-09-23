# How this demo is built

This project is a self-contained Retrieval-Augmented Generation application
built on Spring Boot 3 and langchain4j 0.34, talking to a local Ollama
server. It serves a single-page web UI whose answers stream token by token
over Server-Sent Events (SSE).

## Components

- **OllamaStreamingChatModel** - langchain4j's streaming chat model for the
  local Ollama server, configured with llama3.2:1b.
- **RagAssistant** - a one-method interface implemented at runtime by
  langchain4j AiServices. Its `@UserMessage` template injects the retrieved
  chunks as a `{{context}}` variable; the guardrail rules live in the system
  message. Retrieval is owned by the application, so the answer is grounded
  in exactly the chunks the UI shows.
- **TfIdfEmbeddingModel** - a custom langchain4j EmbeddingModel implementing
  a lexical TF-IDF vector space over the ingested corpus. It needs no extra
  model download, which is why the whole demo runs with a single
  `ollama pull llama3.2:1b`. It can be swapped for an Ollama embedding model
  (nomic-embed-text) by setting `rag.embedding.provider=ollama`.
- **InMemoryEmbeddingStore** - langchain4j's built-in vector store; each
  stored entry is a TextSegment with file name and chunk index metadata.
- **RagService retrieval rules** - the single retrieval leg of the pipeline:
  embed the question, take the top-K cosine matches from the embedding
  store, then apply the absolute minimum-score floor and the relative
  margin. The surviving matches feed both the UI source cards and the
  prompt context, so the two can never diverge.
- **KnowledgeBaseService** - startup ingestion: loads .md/.txt documents
  from an external folder or the bundled classpath folder, splits them with
  DocumentSplitters.recursive (700 characters, 120 overlap), fits the TF-IDF
  vocabulary, embeds every segment and fills the embedding store.
- **ChatController / RagService** - REST + SSE endpoints. The chat endpoint
  sends a `sources` event first (files, scores, snippets), then `token`
  events as the model generates, then a `done` event with timings.

## Answering a question, step by step

1. The browser opens an EventSource to `/api/chat/stream?question=...`.
2. RagService embeds the question with the same model used at ingestion
   time. If the query shares no vocabulary with the corpus (a zero vector),
   no sources are returned at all.
3. The embedding store ranks every segment by cosine similarity. Two
   thresholds then apply: chunks under the absolute minimum score are
   dropped (guarding against "everything is weak"), and chunks scoring far
   below the best hit are dropped by a relative margin, so a strong match is
   not diluted by noise. The survivors are sent to the browser first.
4. AiServices renders the prompt: the system guardrails plus a user message
   carrying the retrieved context excerpts (with their source files) and the
   question.
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
