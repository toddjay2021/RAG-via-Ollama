# Embeddings and similarity search

An embedding is a fixed-length numeric vector that represents the meaning of
a piece of text. Texts that talk about similar things map to vectors that
point in similar directions, so "how to index documents" and "document
ingestion" end up close together even though they share few words.

## Two families of embeddings

- **Lexical (statistical)** methods such as TF-IDF build a vocabulary from the
  corpus and weight terms by how distinctive they are. They excel at exact
  terminology matches, need no model at all, and are fully explainable.
- **Semantic (neural)** embedding models such as nomic-embed-text or
  all-MiniLM are trained to capture meaning, so they also match paraphrases
  and synonyms, at the cost of a model download and inference time.

## TF-IDF in one paragraph

TF-IDF stands for Term Frequency times Inverse Document Frequency. A term
that appears often *in one chunk* but rarely *across the corpus* receives a
high weight; terms that appear everywhere (like common words) receive almost
none. Each chunk becomes a sparse vector over the corpus vocabulary. This
demo builds its vocabulary from the knowledge base itself during ingestion.

## Cosine similarity

Cosine similarity measures the angle between two vectors, ignoring their
length. It ranges from -1 to 1, where 1 means identical direction. Retrieval
ranks all stored chunk vectors by their cosine similarity to the query vector
and returns the top K results. A threshold on the score filters out chunks
that are not relevant enough.

## Choosing a vector store

For a few hundred chunks an in-memory linear scan answers instantly, which
is what this demo does. Production systems with millions of vectors use
dedicated stores such as Qdrant, pgvector, Milvus, or Chroma that index
vectors with approximate-nearest-neighbor algorithms such as HNSW.
