"""Evaluate retrieval + answer quality and run the ablation (no-RAG vs RAG vs RAG+rerank).

Usage:  python eval/evaluate.py --questions eval/questions.json --k 5
Outputs: eval/results.csv (per question) and eval/results.md (summary tables for the PPT)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chat import RAGChat  # noqa: E402
from src.generator import LLM  # noqa: E402
from src.retriever import Retriever  # noqa: E402


def is_refusal(ans: str) -> bool:
    return "could not find" in ans.lower()


def is_correct(ans: str, keywords) -> bool:
    a = ans.lower().replace(",", "")
    return all(k.lower().replace(",", "") in a for k in keywords)


def cited(ans: str, sources):
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", ans)}
    return [(s["company"].lower(), s["page"]) for s in sources if s["n"] in nums]


def retrieval_metrics(retriever, qs, k, rerank):
    hits, rr = [], []
    for q in qs:
        res = retriever.search(q["question"], top_k=k, company=q.get("company"), rerank=rerank)
        gold = set(q["expected_pages"])
        rank = next((i + 1 for i, r in enumerate(res) if r["page"] in gold), None)
        hits.append(1 if rank else 0)
        rr.append(1 / rank if rank else 0)
    n = len(qs)
    return {"Hit@k": sum(hits) / n, "MRR": sum(rr) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--index", default="index")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    qs = [q for q in json.loads(Path(a.questions).read_text()) if not q.get("_note")]
    retriever = Retriever(a.index)
    bot = RAGChat(retriever, LLM())

    # 1) Retrieval quality
    ret_rows = []
    for name, rerank in [("Dense", False), ("Dense + rerank", True)]:
        m = retrieval_metrics(retriever, qs, a.k, rerank)
        ret_rows.append({"Retriever": name, **m})
    ret_df = pd.DataFrame(ret_rows)

    # 2) Answer quality ablation
    modes = {
        "LLM only (no RAG)": dict(use_rag=False),
        "RAG": dict(use_rag=True, rerank=False),
        "RAG + reranker": dict(use_rag=True, rerank=True),
    }
    rows = []
    for mode, kw in modes.items():
        for q in qs:
            bot.reset()
            res = bot.ask(q["question"], top_k=a.k, company=q.get("company"), use_memory=False, **kw)
            ans = res["answer"]
            correct = is_correct(ans, q["answer_keywords"])
            refused = is_refusal(ans)
            gold = {(q.get("company", "").lower(), p) for p in q["expected_pages"]}
            cite_ok = bool(set(cited(ans, res["sources"])) & gold) if kw["use_rag"] else None
            rows.append({
                "mode": mode, "question": q["question"], "answer": ans,
                "correct": correct, "refused": refused,
                "hallucinated": (not refused) and (not correct),
                "citation_ok": cite_ok,
            })
    df = pd.DataFrame(rows)
    df.to_csv("eval/results.csv", index=False)

    summary = df.groupby("mode").agg(
        Accuracy=("correct", "mean"),
        Hallucination_rate=("hallucinated", "mean"),
        Refusal_rate=("refused", "mean"),
        Citation_accuracy=("citation_ok", lambda s: s.dropna().mean() if s.notna().any() else float("nan")),
    ).reindex(list(modes)).round(3)

    md = "## Retrieval\n\n" + ret_df.round(3).to_markdown(index=False)
    md += "\n\n## Answer quality (ablation)\n\n" + summary.reset_index().to_markdown(index=False)
    Path("eval/results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
