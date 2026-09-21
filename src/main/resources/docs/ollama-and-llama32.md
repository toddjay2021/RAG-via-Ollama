# Ollama and the llama3.2:1b model

Ollama is a local runtime for large language models. It downloads GGUF model
weights, exposes a small REST API on `http://localhost:11434`, and keeps
models resident in memory so consecutive requests are fast. Everything runs
on the local machine, so no prompt or document ever leaves it.

## Core API endpoints

| Endpoint      | Purpose                                                  |
|---------------|----------------------------------------------------------|
| `GET /api/tags`   | list installed models                              |
| `POST /api/chat`  | chat completion; supports NDJSON streaming           |
| `POST /api/embed` | compute embedding vectors with an embedding model   |

Models are managed with the `ollama pull <model>` command. This demo uses
`llama3.2:1b`, a compact 1.2-billion-parameter model that runs comfortably
on ordinary laptops while still following grounded-answer instructions.

## llama3.2:1b characteristics

- 1.2B parameters, roughly a 1.3 GB download in Q8_0 quantization.
- 128k token context length, more than enough for retrieved chunks plus a question.
- Suitable for CPU inference; a small GPU makes it faster still.
- Multilingual instruction-following quality good enough for demos and prototyping.

## Why a small model is enough for RAG

In a retrieval-augmented pipeline the heavy lifting of *knowledge* is done by
retrieval; the model only has to reformulate the provided evidence into an
answer. That is a much easier task than recalling facts from parameters, so
a 1B model with good context grounding often matches far larger models on
knowledge-base questions, at a fraction of the cost and latency.
