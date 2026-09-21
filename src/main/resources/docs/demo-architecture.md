# How this demo is built

This project is a self-contained Retrieval-Augmented Generation application
written in Java 17 with a single runtime dependency (Gson). It talks to a
local Ollama server and serves two interfaces: an interactive command line
and a small web UI with a JSON API.

## Components

- **OllamaClient** - minimal HTTP client for the Ollama REST API: model
  listing, streamed chat completions, and embeddings.
- **DocumentChunker** - paragraph- and sentence-aware splitting with
  configurable size and overlap.
- **EmbeddingProvider** - pluggable embedding layer. The default `tfidf`
  provider builds a lexical vector space from the corpus with zero extra
  downloads; the `ollama` provider calls the Ollama embedding endpoint for
  semantic vectors.
- **VectorStore** - in-memory store with cosine-similarity top-K search.
- **RagPipeline** - orchestrates ingestion (load, chunk, fit, embed, store)
  and serving (embed question, retrieve, build augmented prompt, stream the
  grounded answer).
- **WebServer** - the JDK built-in HTTP server exposing `/`, `/api/health`,
  `/api/retrieve` and `/api/answer`, the latter streaming tokens as they are
  produced.

## Answering a question, step by step

1. The user submits a question via the CLI or the web UI.
2. The question is embedded with the same provider used at ingestion time.
3. The vector store ranks every chunk by cosine similarity and returns the
   top 3 above a minimum score.
4. A prompt is assembled: a system prompt with grounding rules, the numbered
   context excerpts with their source files, and the question.
5. llama3.2:1b generates the answer, which streams token by token back to
   the interface, together with the list of sources and similarity scores.

## Configuration

All knobs live in `app.properties` (model name, embedding provider, chunk
size, overlap, top K, minimum score, temperature, web port) and every value
can be overridden with an environment variable such as `OLLAMA_MODEL` or
`EMBEDDING_PROVIDER`.

## Swapping parts

The interfaces are intentionally small so each part can be replaced for
production: a persistent vector database instead of the in-memory store,
nomic-embed-text instead of TF-IDF, a larger model such as llama3.2:3b, or
a Spring Boot layer around the same `RagPipeline` class.
