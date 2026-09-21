# Chunking strategies

Chunking is the step that splits long documents into pieces small enough to
embed and retrieve individually. It is the most underestimated part of a RAG
pipeline: bad chunk boundaries destroy retrieval quality no matter how good
the embedding model is.

## Why chunks are needed

Embedding models compress meaning into a fixed-size vector, and prompts have
a context limit. Very large text dilutes the signal - a vector of an entire
book answers nothing specifically. Very small fragments lose the context
that gives words their meaning. The goal is chunks that are **self-contained
enough to be understood alone but focused enough to be about one topic**.

## Practical defaults

- 300 to 800 characters (or roughly 100-200 tokens) is a common starting
  window for short-form documents such as FAQs and manuals.
- A 10 to 20 percent overlap between consecutive chunks keeps sentences that
  straddle a boundary retrievable from both sides.
- Splitting on natural boundaries - paragraphs first, then sentences - beats
  cutting at arbitrary character offsets, because semantic units stay whole.

## How this demo chunks

The demo uses a hybrid strategy: it splits on blank lines (paragraphs), then
on sentence boundaries for oversized paragraphs, then greedily packs the
resulting units into chunks of at most 700 characters, carrying a 120
character tail into the next chunk as overlap. The defaults are configurable
via `chunk.size.chars` and `chunk.overlap.chars`.

## Failure modes to watch

- A table or list cut in half becomes unretrievable noise.
- Overly aggressive overlap wastes context budget on repeated text.
- Fixed windows that ignore structure turn coherent prose into fragments
  that begin and end mid-thought, confusing both the retriever and the model.
