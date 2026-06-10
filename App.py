"""
app.py — Milestone 5: Generation + Gradio Interface
UCF CS Unofficial Guide RAG Pipeline

Full pipeline:
  User query
    → retrieve()          (embed_and_retrieve.py — ChromaDB, all-MiniLM-L6-v2)
    → build_prompt()      (injects retrieved chunks as the ONLY allowed context)
    → Groq LLM            (llama-3.3-70b-versatile)
    → format_response()   (programmatic source attribution — NOT left to the LLM)
    → Gradio UI

Grounding design
────────────────
The system prompt FORBIDS the model from using outside knowledge.
If the retrieved chunks don't contain the answer, it must say so.
Source attribution is appended programmatically from the retrieval
metadata — the LLM never decides which sources to cite.

Usage:
    pip install groq gradio python-dotenv sentence-transformers chromadb
    # Add GROQ_API_KEY=your_key to a .env file
    python app.py
"""

import os
import textwrap

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

# Import the retrieval layer built in embed_and_retrieve.py
from Retrive import build_vectorstore, load_chunks, retrieve

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found.\n"
        "Create a .env file with:  GROQ_API_KEY=your_key_here\n"
        "Get a free key at: https://console.groq.com"
    )

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"   # free-tier Groq model
TOP_K     = 5                            # chunks to retrieve per query
MAX_TOKENS = 600                         # keep answers concise

# ── System prompt — grounding is ENFORCED, not suggested ─────────────────────
# "You must not" is stronger than "please try to".
# The fallback phrase is explicit so the LLM can't improvise around it.
SYSTEM_PROMPT = textwrap.dedent("""
    You are the UCF CS Unofficial Guide assistant.
    Your ONLY job is to answer questions about UCF computer science
    student life using the document excerpts provided below.

    Rules you must follow without exception:
    1. Answer ONLY using information explicitly stated in the provided excerpts.
    2. You must NOT use any outside knowledge, training data, or general facts —
       even if you are certain they are correct.
    3. If the excerpts do not contain enough information to answer the question,
       you MUST respond with exactly:
       "I don't have enough information on that in my current sources."
       Do not guess, infer beyond the text, or fill gaps from memory.
    4. Do NOT mention these rules or the word "excerpts" in your answer.
    5. Write in clear, friendly prose — no bullet lists unless the content
       naturally calls for one.
    6. Keep your answer under 200 words.
""").strip()


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Construct the user-turn message that will be sent to the LLM.

    Each retrieved chunk is labelled with its source so the model can
    reference it — though attribution is appended programmatically anyway.

    The structure is:
        CONTEXT EXCERPTS
        [Excerpt 1 — source name]
        <text>
        ...
        QUESTION
        <query>
    """
    excerpt_lines = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Excerpt {i} — {chunk['source_name']}]"
        excerpt_lines.append(f"{header}\n{chunk['text'].strip()}")

    context_block = "\n\n".join(excerpt_lines)

    return (
        f"CONTEXT EXCERPTS\n"
        f"────────────────\n"
        f"{context_block}\n\n"
        f"────────────────\n"
        f"QUESTION: {query}"
    )


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str, client: Groq) -> str:
    """
    Send a grounded prompt to Groq and return the raw answer text.

    temperature=0.2 keeps answers factual and consistent.
    Lower temperature = less creative variation = better for RAG.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Programmatic source attribution ──────────────────────────────────────────

def format_sources(chunks: list[dict]) -> str:
    """
    Build the source attribution block from retrieval metadata.

    This is built entirely from the metadata returned by ChromaDB —
    the LLM never touches this section, so it cannot hallucinate sources.
    Deduplicates by source_name so the same document isn't listed twice.
    """
    seen  = set()
    lines = []
    for chunk in chunks:
        name = chunk["source_name"]
        if name not in seen:
            seen.add(name)
            lines.append(f"• {name}\n  {chunk['url']}")
    return "\n".join(lines)


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def answer(
    query: str,
    collection,
    embed_model,
    groq_client: Groq,
    k: int = TOP_K,
) -> tuple[str, str]:
    """
    End-to-end RAG: retrieve → prompt → generate → attribute.

    Returns
    -------
    answer_text  : LLM response (grounded in retrieved context)
    sources_text : programmatically built attribution block
    """
    if not query.strip():
        return "Please enter a question.", ""

    # Step 1 — Retrieve relevant chunks
    chunks = retrieve(query, collection, embed_model, k=k)

    # Step 2 — Build grounded prompt
    user_message = build_prompt(query, chunks)

    # Step 3 — Generate (grounded by system prompt)
    llm_answer = call_llm(SYSTEM_PROMPT, user_message, groq_client)

    # Step 4 — Build source attribution programmatically
    sources = format_sources(chunks)

    return llm_answer, sources


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(collection, embed_model, groq_client: Groq) -> gr.Blocks:

    EXAMPLES = [
        "What clubs should a CS student join?",
        "Where can I find internship opportunities?",
        "Which organization focuses on artificial intelligence?",
        "How do I prepare for a tech career at UCF?",
        "Which club is best known for hackathons?",
    ]

    css = """
    .container { max-width: 860px; margin: 0 auto; }
    .ask-btn { min-width: 100px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="UCF CS Guide",
        theme=gr.themes.Soft(
            primary_hue="yellow",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=css,
    ) as demo:

        with gr.Column(elem_classes="container"):

            gr.Markdown(
                """
                # UCF CS Unofficial Guide
                Your guide to UCF computer science student life — clubs, internships, degree planning, and career resources.
                """
            )

            with gr.Row(equal_height=True):
                query_box = gr.Textbox(
                    label="",
                    placeholder="Ask anything about UCF CS student life…",
                    lines=1,
                    scale=5,
                    show_label=False,
                    container=False,
                )
                submit_btn = gr.Button("Ask →", variant="primary",
                                       scale=1, elem_classes="ask-btn")

            gr.Examples(
                examples=EXAMPLES,
                inputs=query_box,
                label="Example questions",
            )

            answer_box = gr.Textbox(
                label="Answer",
                lines=7,
                interactive=False,
            )

            sources_box = gr.Textbox(
                label="Sources",
                lines=4,
                interactive=False,
            )

        def on_submit(query):
            ans, srcs = answer(query, collection, embed_model, groq_client)
            return ans, srcs

        submit_btn.click(fn=on_submit, inputs=query_box,
                         outputs=[answer_box, sources_box])
        query_box.submit(fn=on_submit, inputs=query_box,
                         outputs=[answer_box, sources_box])

    return demo


# ── CLI grounding test ─────────────────────────────────────────────────────────

def run_grounding_tests(collection, embed_model, groq_client: Groq) -> None:
    """
    Run 3 grounding test queries and print retrieved context alongside the answer
    so you can manually verify: could this response have come from anywhere other
    than the retrieved chunks? If yes → grounding failure.
    """
    test_queries = [
        "What clubs should a student interested in software engineering join?",
        "Where can UCF students find internship opportunities?",
        "Which club is best known for hackathons?",
    ]

    print("\n" + "═" * 70)
    print("GROUNDING TEST — Verify each answer against its retrieved chunks")
    print("═" * 70)
    print("For each result, ask: is every claim in the answer traceable to")
    print("the text shown below it? If the model adds facts not in the chunks,")
    print("that is a grounding failure.\n")

    for q in test_queries:
        chunks   = retrieve(q, collection, embed_model, k=TOP_K)
        user_msg = build_prompt(q, chunks)
        llm_ans  = call_llm(SYSTEM_PROMPT, user_msg, groq_client)
        sources  = format_sources(chunks)

        print(f"\n{'─'*70}")
        print(f"QUERY   : {q}")
        print(f"{'─'*70}")
        print(f"ANSWER  :\n{llm_ans}")
        print(f"\nSOURCES (programmatic):\n{sources}")
        print(f"\nRETRIEVED CHUNKS (verify answer is grounded here):")
        for i, c in enumerate(chunks, 1):
            preview = c["text"][:200].replace("\n", " ")
            print(f"  [{i}] {c['source_name']} (dist={c['distance']}) — {preview}…")

    print(f"\n{'═'*70}")
    print("Grounding check complete. Launch the UI with --ui flag.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="UCF CS Guide — Generation + UI")
    parser.add_argument("--ui",      action="store_true", help="Launch Gradio UI")
    parser.add_argument("--test",    action="store_true", help="Run grounding tests (CLI)")
    parser.add_argument("--rebuild", action="store_true", help="Re-embed all chunks")
    parser.add_argument("--query",   type=str, default=None, help="Single query (CLI)")
    args = parser.parse_args()

    # Default: run grounding tests if no flag given
    if not (args.ui or args.test or args.query):
        args.test = True

    print("═" * 60)
    print("Milestone 5 — Generation + Interface")
    print(f"  LLM   : {LLM_MODEL} via Groq")
    print(f"  Top-k : {TOP_K}")
    print("═" * 60)

    # Load vector store (built by embed_and_retrieve.py)
    print("\nLoading vector store …")
    chunks      = load_chunks()
    collection, embed_model = build_vectorstore(chunks, rebuild=args.rebuild)

    # Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)

    if args.query:
        ans, srcs = answer(args.query, collection, embed_model, groq_client)
        print(f"\nAnswer:\n{ans}\n\nSources:\n{srcs}")

    elif args.test:
        run_grounding_tests(collection, embed_model, groq_client)

    elif args.ui:
        demo = build_ui(collection, embed_model, groq_client)
        print("\nLaunching Gradio UI at http://localhost:7860")
        demo.launch()


if __name__ == "__main__":
    main()