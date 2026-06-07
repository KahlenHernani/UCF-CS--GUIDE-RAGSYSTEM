"""
ingest.py — Milestone 3: Document Ingestion & Chunking
UCF CS Unofficial Guide RAG Pipeline

Loads documents from the 10 sources in planning.md, cleans the text,
and splits into chunks (size=500 chars, overlap=100 chars) using
RecursiveCharacterTextSplitter. Saves chunks to chunks.json.

Usage:
    pip install requests beautifulsoup4 langchain-text-splitters praw
    python ingest.py
"""

import json
import re
import time
import requests
import praw
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config (matches planning.md) ─────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

SOURCES = [
    {
        "id": 1,
        "name": "UCF Computer Science BS Program",
        "type": "web",
        "url": "https://www.ucf.edu/degree/computer-science-bs/",
    },
    {
        "id": 2,
        "name": "UCF Computer Science Department",
        "type": "web",
        "url": "https://www.cs.ucf.edu/",
    },
    {
        "id": 3,
        "name": "UCF CS Student Organizations",
        "type": "web",
        "url": "https://www.cs.ucf.edu/student-organizations/",
    },
    {
        "id": 4,
        "name": "CECS Student Organizations",
        "type": "web",
        "url": "https://www.cecs.ucf.edu/current-students/student-organizations/",
    },
    {
        "id": 5,
        "name": "ACM@UCF",
        "type": "web",
        "url": "https://ucf.acm.org/",
    },
    {
        "id": 6,
        "name": "Knight Hacks",
        "type": "web",
        "url": "https://knighthacks.org/",
    },
    {
        "id": 7,
        "name": "KnightConnect / Get Involved",
        "type": "web",
        "url": "https://osi.ucf.edu/registered-student-organizations-rsos/get-involved/",
    },
    {
        "id": 8,
        "name": "Dixon Career Development Center",
        "type": "web",
        "url": "https://career.ucf.edu/",
    },
    {
        "id": 9,
        "name": "UCF Handshake",
        "type": "web",
        "url": "https://career.ucf.edu/resources/handshake/",
    },
    {
        "id": 10,
        "name": "r/UCF Reddit Community",
        "type": "reddit",
        "subreddit": "ucf",
        # Fetches top posts + top comments so the content is substantive
        "limit": 40,
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and strip boilerplate noise."""
    # Collapse runs of whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Drop lines that are clearly nav / cookie / JS noise (short, no sentence)
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 20 or l == ""]
    return "\n".join(lines).strip()


def fetch_web(url: str) -> str:
    """Download a page and return its visible text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script / style / nav / footer noise
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "form"]):
        tag.decompose()

    return clean_text(soup.get_text(separator="\n"))


def fetch_reddit(subreddit: str, limit: int) -> str:
    """
    Pull top posts + their top comments from a subreddit using the
    public JSON API (no credentials required).
    Falls back gracefully if rate-limited.
    """
    texts = []
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t=year"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        posts = resp.json()["data"]["children"]
    except Exception as e:
        print(f"  [WARN] Reddit fetch failed: {e}")
        return ""

    for post in posts:
        d = post["data"]
        title   = d.get("title", "")
        selftext = d.get("selftext", "")
        texts.append(f"POST: {title}\n{selftext}")

        # Fetch top-level comments for each post
        comments_url = (
            f"https://www.reddit.com/r/{subreddit}/comments/"
            f"{d['id']}.json?limit=10&sort=top"
        )
        try:
            cr = requests.get(comments_url, headers=HEADERS, timeout=10)
            cr.raise_for_status()
            comment_listing = cr.json()[1]["data"]["children"]
            for c in comment_listing[:5]:
                body = c["data"].get("body", "")
                if body and body != "[deleted]":
                    texts.append(f"COMMENT: {body}")
        except Exception:
            pass  # comments are a bonus; silently skip on error

        time.sleep(0.6)  # be polite to Reddit's rate limit

    return clean_text("\n\n".join(texts))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest_all() -> list[dict]:
    """Load all sources and return a list of raw document dicts."""
    docs = []
    for src in SOURCES:
        print(f"[{src['id']:02d}] Fetching: {src['name']} ...")
        if src["type"] == "web":
            text = fetch_web(src["url"])
        elif src["type"] == "reddit":
            text = fetch_reddit(src["subreddit"], src["limit"])
        else:
            print(f"  [SKIP] Unknown source type: {src['type']}")
            continue

        if not text:
            print(f"  [WARN] Empty document — skipping source {src['id']}")
            continue

        docs.append({
            "source_id":   src["id"],
            "source_name": src["name"],
            "url":         src.get("url", f"r/{src.get('subreddit','')}"),
            "text":        text,
        })
        print(f"  → {len(text):,} chars extracted")

    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split each document into overlapping chunks using
    RecursiveCharacterTextSplitter (size=500, overlap=100).
    Returns a flat list of chunk dicts ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Try to split on paragraph → sentence → word → char boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in docs:
        raw_chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id":    f"src{doc['source_id']:02d}_chunk{i:04d}",
                "source_id":   doc["source_id"],
                "source_name": doc["source_name"],
                "url":         doc["url"],
                "text":        chunk_text,
                "char_count":  len(chunk_text),
            })

    return all_chunks


def main():
    print("=" * 60)
    print("UCF CS Unofficial Guide — Ingestion & Chunking")
    print(f"  Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print("=" * 60)

    # 1. Ingest
    docs = ingest_all()
    print(f"\n✓ Loaded {len(docs)} documents\n")

    # 2. Chunk
    chunks = chunk_documents(docs)
    print(f"✓ Produced {len(chunks)} chunks total")

    # 3. Quick sanity check
    sizes = [c["char_count"] for c in chunks]
    print(f"  Chunk sizes → min: {min(sizes)}, max: {max(sizes)}, "
          f"avg: {sum(sizes)//len(sizes)}")

    # 4. Save
    output_path = "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved chunks to {output_path}")

    # 5. Preview first chunk
    print("\n── Sample chunk ──────────────────────────────────────────")
    sample = chunks[0]
    print(f"  ID:     {sample['chunk_id']}")
    print(f"  Source: {sample['source_name']}")
    print(f"  Chars:  {sample['char_count']}")
    print(f"  Text:   {sample['text'][:200]}...")
    print("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()