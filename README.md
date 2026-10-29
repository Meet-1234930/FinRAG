# FinRAG: Source-Grounded Conversational Assistant for Indian Company Filings

Course: Deep Learning and GenAI (CS312B), CA-2, Track B (Transformers and LLM pipelines)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indian company annual
reports and earnings call transcripts, **cites the exact company / year / page** for every answer,
and remembers the conversation for follow-up questions.

## Architecture

```
PDFs / transcripts -> PyMuPDF text (per page) -> word chunks (300 words, 60 overlap)
   -> bge-small embeddings -> FAISS index

Question (+ last 5 turns) -> LLM query rewrite (standalone question)
   -> FAISS top-30 (optional company filter) -> optional cross-encoder rerank -> top-k
   -> Prompt: numbered context + "answer only from context, cite [n]"
   -> Qwen2.5-3B-Instruct (4-bit NF4) -> answer + source list
```

| Component | Choice |
|---|---|
| LLM | Qwen/Qwen2.5-3B-Instruct, 4-bit (bitsandbytes). Override with `FINRAG_MODEL` env var |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, normalized) |
| Vector store | FAISS `IndexFlatIP` (cosine) |
| Reranker (optional) | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Memory | Window buffer of last 5 turns, used for query rewriting |
| UI | Gradio |

## Setup (Google Colab, free T4 GPU)

1. Runtime -> Change runtime type -> T4 GPU.
2. Upload this folder (or clone the repo), then:
   ```bash
   pip install -r requirements.txt
   ```
3. Add PDFs to `data/` named `Company_Year.pdf` (see `data/README.txt`).
4. Build the index:
   ```bash
   python -m src.ingest --data data --out index
   ```
5. Launch the demo (prints a public Gradio link):
   ```bash
   python app.py
   ```
6. Evaluate (after filling `eval/questions.json` with your own questions):
   ```bash
   python eval/evaluate.py --questions eval/questions.json --k 5
   ```
   Writes `eval/results.csv` and `eval/results.md` (tables to paste into the report).

`notebook.ipynb` runs all of the above in order.

## Evaluation protocol

- **Test set:** 30-40 hand-written questions with gold page numbers and answer keywords.
- **Retrieval:** Hit@k and MRR (is a gold page in the top-k?).
- **Answer quality:** accuracy (all gold keywords present), hallucination rate (confident but wrong),
  refusal rate, citation accuracy (cited passage is a gold page).
- **Ablation:** LLM only vs RAG vs RAG + reranker.

Keyword matching is a cheap proxy; spot-check a sample manually and report that too.

## Known limitations

- Tables in PDFs are flattened by text extraction, so numeric questions over tables can fail.
- Scanned (image-only) PDFs need OCR; they are not supported.
- A 3B model can still misread a correct passage; the reranker and larger `top_k` help partly.

## Repository layout

```
app.py              Gradio demo
src/ingest.py       PDF -> chunks -> embeddings -> FAISS
src/retriever.py    dense retrieval + rerank + company detection
src/generator.py    4-bit LLM wrapper
src/chat.py         RAG pipeline with memory and citations
eval/evaluate.py    metrics + ablation
eval/questions.json test set (template)
notebook.ipynb      one-click Colab run
```


