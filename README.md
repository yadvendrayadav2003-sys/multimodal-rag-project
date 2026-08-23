# Multimodal RAG with LangChain + Gemini

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yadvendrayadav2003-sys/multimodal-rag-project/blob/main/langchain_multimodal_gemini.ipynb)(MY_NOTEBOOK_GITHUB_LINK)

A Retrieval-Augmented Generation (RAG) pipeline that understands **text, tables, and images**
inside a PDF. It parses a document, generates AI summaries of every element (including images),
embeds those summaries into a vector store, and answers natural-language questions by retrieving
and reasoning over the original text, tables, *and* images — not just plain text.

Runs end-to-end in **Google Colab** using **Google Gemini** for both the language model and the
embeddings, so it only needs a single free API key.

## How it works

1. **Extract** — [`unstructured`](https://docs.unstructured.io/) partitions the PDF into text
   chunks, tables, and images (as base64).
2. **Summarize** — Each text chunk, table, and image is summarized by Gemini
   (`gemini-flash-lite-latest`), so everything — including images — becomes searchable text.
3. **Index** — Summaries are embedded (`gemini-embedding-001`) and stored in a
   [Chroma](https://www.trychroma.com/) vector store. The original full content (raw text, table
   HTML, image bytes) is kept in a simple lookup so retrieval can return the *source* material,
   not just the summary.
4. **Retrieve & Answer** — A question is embedded, matched against summaries, and the
   corresponding original text/tables/images are passed — images included — to Gemini to generate
   a grounded answer.

## Tech stack

| Purpose              | Tool                                         |
|-----------------------|-----------------------------------------------|
| PDF parsing            | `unstructured[all-docs]`                     |
| Orchestration          | LangChain (`langchain`, `langchain-core`)     |
| LLM + Vision + Embeddings | Google Gemini (`langchain-google-genai`)   |
| Vector store            | Chroma (`langchain-chroma`)                  |
| Runtime                 | Google Colab                                 |

## Getting started

### 1. Get a Gemini API key
Create a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Open the notebook in Colab
Upload `langchain_multimodal_gemini.ipynb` to [Google Colab](https://colab.research.google.com/).

### 3. Add your API key
Either:
- **Recommended:** In Colab, click the 🔑 **Secrets** icon in the left sidebar → add a new secret
  named `GOOGLE_API_KEY` with your key as the value → toggle **Notebook access** on.
- **Or:** just run the notebook — it will prompt you to paste the key manually if no secret is found.

### 4. Run all cells
`Runtime → Run all`. The notebook downloads a sample PDF (the "Attention Is All You Need" paper)
automatically, so no file upload is required.

> **Note on speed:** the summarization steps are deliberately paced (~5s between API calls) to
> stay within Gemini's free-tier rate limits. Expect the summarization cells to take a few minutes
> for a paper-sized PDF.

### 5. Ask your own questions
At the bottom of the notebook, change the question passed to `chain.invoke(...)` or
`chain_with_sources.invoke(...)` to ask anything about the PDF's content.

## Using your own PDF

Replace the download step with your own file:
```python
file_path = "./content/your_file.pdf"
# upload it to Colab first (left sidebar → Files → upload), then re-run partition_pdf
```

## Notes on the free tier

This project intentionally uses `gemini-flash-lite-latest` — a **stable alias** that Google
automatically points to their current recommended lightweight Flash model, rather than a
dated model name (e.g. `gemini-2.5-flash-lite`) that can get deprecated for new API keys.
If you hit a `429 RESOURCE_EXHAUSTED` error, you've likely hit your daily/per-minute quota —
check your current limits at [aistudio.google.com/usage](https://aistudio.google.com/usage) and
either wait a bit or upgrade to a paid tier.

## Acknowledgements

- Adapted from [Alejandro AO's](https://github.com/alejandro-ao) original OpenAI-based multimodal
  RAG tutorial, ported to run entirely on Gemini.
- [LangChain multi-modal RAG cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb)
- [LangChain Multi-Vector Retriever guide](https://python.langchain.com/docs/how_to/multi_vector/)

## License

MIT
