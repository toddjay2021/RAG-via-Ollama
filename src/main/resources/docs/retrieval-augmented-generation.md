# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is an architecture that combines a language
model with an external knowledge source. Instead of relying only on what the
model memorized during training, the system first *retrieves* relevant
documents and then *generates* an answer grounded in them.

## Why RAG exists

Large language models have three well-known problems that RAG addresses:

1. **Knowledge cutoff** - a model cannot answer questions about events or
   facts that appeared after its training data was collected.
2. **Hallucination** - when a model does not know an answer it often produces
   a confident but wrong response instead of admitting ignorance.
3. **Private data** - organizations want answers based on their own manuals,
   tickets, or wikis, which were never part of any training set.

Retrieval solves all three: the knowledge base is fresh (update the
documents, not the model), answers are grounded in real excerpts, and the
corpus can contain anything the operator chooses.

## The pipeline

A RAG system has two phases:

- **Ingestion (offline)**: documents are loaded, split into chunks, converted
  to vectors (embeddings), and stored in a vector store.
- **Serving (online)**: a user question is embedded, the most similar chunks
  are retrieved with a similarity metric such as cosine similarity, and the
  chunks are inserted into a prompt that is sent to the model.

Because the model receives the actual evidence in its context window, it can
quote, summarize, and cite the source instead of guessing. A good system
prompt also instructs the model to say when the knowledge base does not
contain the answer, which further reduces hallucination.

## Where it is used

Typical production uses include enterprise FAQ and documentation assistants,
support-ticket copilots, legal and medical document Q&A with citations,
codebase assistants, and any chatbot that must answer from a curated corpus
rather than the open internet.
