"""Gradio demo:  python app.py  (set share=True for a public Colab link)."""
import gradio as gr

from src.chat import RAGChat, format_sources
from src.generator import LLM
from src.retriever import Retriever

retriever = Retriever("index")
llm = LLM()
bot = RAGChat(retriever, llm)

COMPANIES = ["All"] + retriever.companies


def respond(message, history, company, top_k, rerank, use_rag):
    history = history or []
    res = bot.ask(
        message,
        use_rag=use_rag,
        rerank=rerank,
        top_k=int(top_k),
        company=None if company == "All" else company,
    )
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": res["answer"]},
    ]
    info = f"*Standalone query used for retrieval:* `{res['standalone_question']}`\n\n" + format_sources(res["sources"])
    return history, info, ""


def clear():
    bot.reset()
    return [], "", ""


with gr.Blocks(title="FinRAG") as demo:
    gr.Markdown("# FinRAG\nSource-grounded Q&A over Indian company annual reports and transcripts")
    with gr.Row():
        with gr.Column(scale=3):
            try:
                chatbot = gr.Chatbot(type="messages", height=480)
            except TypeError:  # newer Gradio: messages format is the default
                chatbot = gr.Chatbot(height=480)
            msg = gr.Textbox(placeholder="e.g. What was TCS's revenue growth in FY24?", label="Question")
            with gr.Row():
                send = gr.Button("Ask", variant="primary")
                clr = gr.Button("Clear chat")
        with gr.Column(scale=2):
            company = gr.Dropdown(COMPANIES, value="All", label="Company filter")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-k passages")
            rerank = gr.Checkbox(value=False, label="Cross-encoder reranking")
            use_rag = gr.Checkbox(value=True, label="Use RAG (untick = plain LLM baseline)")
            sources = gr.Markdown("Sources will appear here.")

    inputs = [msg, chatbot, company, top_k, rerank, use_rag]
    outputs = [chatbot, sources, msg]
    msg.submit(respond, inputs, outputs)
    send.click(respond, inputs, outputs)
    clr.click(clear, None, outputs)

if __name__ == "__main__":
    demo.launch(share=True)
