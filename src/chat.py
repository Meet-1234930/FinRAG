"""RAG pipeline: query rewrite (memory) -> retrieve -> grounded generation with citations."""
from .generator import LLM
from .retriever import Retriever

REFUSAL = "I could not find this in the provided documents."

SYS_RAG = (
    "You are FinRAG, an assistant for Indian company annual reports and earnings call transcripts. "
    "Answer ONLY from the numbered context passages. After every factual claim add the passage "
    "number in square brackets, like [1] or [2][3]. "
    f"If the context does not contain the answer, reply exactly: {REFUSAL} "
    "Be concise. Copy numbers exactly as written in the context."
)

SYS_PLAIN = (
    "You are an assistant answering questions about Indian listed companies from your own knowledge. "
    "Be concise."
)

SYS_REWRITE = (
    "Rewrite the user's last question as a standalone question using the conversation for context. "
    "Keep company names and years explicit. Output only the rewritten question."
)


def format_context(sources):
    return "\n\n".join(
        f"[{s['n']}] ({s['company']} {s['year']}, page {s['page']})\n{s['text']}" for s in sources
    )


def format_sources(sources):
    if not sources:
        return "_No sources used._"
    lines = []
    for s in sources:
        snippet = s["text"][:220].replace("\n", " ")
        lines.append(f"**[{s['n']}] {s['company']} {s['year']} - page {s['page']}** (score {s['score']:.2f})\n> {snippet}...")
    return "\n\n".join(lines)


class RAGChat:
    def __init__(self, retriever: Retriever, llm: LLM, max_turns: int = 5):
        self.retriever, self.llm, self.max_turns = retriever, llm, max_turns
        self.history = []  # window buffer memory: list of (question, answer)

    def reset(self):
        self.history = []

    def _rewrite(self, question: str) -> str:
        if not self.history:
            return question
        turns = "\n".join(
            f"User: {u}\nAssistant: {a[:300]}" for u, a in self.history[-self.max_turns:]
        )
        msgs = [
            {"role": "system", "content": SYS_REWRITE},
            {"role": "user", "content": f"Conversation:\n{turns}\n\nLast question: {question}\n\nStandalone question:"},
        ]
        out = self.llm.chat(msgs, max_new_tokens=64).strip().split("\n")[0]
        return out or question

    def ask(self, question: str, use_rag: bool = True, rerank: bool = False,
            top_k: int = 5, company: str = None, use_memory: bool = True):
        standalone = self._rewrite(question) if use_memory else question

        if not use_rag:
            msgs = [{"role": "system", "content": SYS_PLAIN}, {"role": "user", "content": standalone}]
            answer = self.llm.chat(msgs)
            sources = []
        else:
            company = company or self.retriever.detect_company(standalone)
            hits = self.retriever.search(standalone, top_k=top_k, company=company, rerank=rerank)
            sources = [{**h, "n": i + 1} for i, h in enumerate(hits)]
            msgs = [
                {"role": "system", "content": SYS_RAG},
                {"role": "user", "content": f"Context:\n{format_context(sources)}\n\nQuestion: {standalone}"},
            ]
            answer = self.llm.chat(msgs)

        if use_memory:
            self.history.append((question, answer))
        return {"answer": answer, "sources": sources, "standalone_question": standalone}
