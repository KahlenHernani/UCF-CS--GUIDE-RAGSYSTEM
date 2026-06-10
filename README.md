# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->
This project focuses on creating an Unofficial Guide to the UCF Computer Science Student Experience. The guide helps students find information about courses, degree planning, clubs, hackathons, internships, career resources, and campus opportunities that are relevant to computer science majors. This knowledge is difficult to find because it is scattered across many sources, including UCF department websites, student organization pages, career services resources, and community discussions on Reddit and Discord. A RAG system can bring these sources together and provide personalized recommendations and answers in one place.
---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | UCF Computer Science BS Program | Official degree requirements, curriculum, and career pathways for CS students. | https://www.ucfedu/degree/computer-science-bs/ |
| 2 | UCF Computer Science Department | Department information, academic resources, and opportunities for CS students. | https://www.csucf.edu/ |
| 3 | UCF CS Student Organizations | Information about ACM, AI@UCF, Knight Hacks, Cyber Defense Club, and Programming Team. | https://wwwcs.ucf.edu/student-organizations/ |
| 4 | CECS Student Organizations | Directory of engineering and technology clubs available to students. | https://www.cecs.ucf.educurrent-students/student-organizations/ |
| 5 | ACM@UCF | Workshops, technical projects, networking events, and professional development opportunities. | https://ucf.acm.org/ |
| 6 | Knight Hacks | Hackathons, mentorship programs, software development projects, and workshops. | https://knighthacks.org/ |
| 7 | KnightConnect | Student organization directory and campus event platform. | https://osi.ucf.eduregistered-student-organizations-rsos/get-involved/ |
| 8 |Dixon Career Development Center|Career advising, internship preparation, career fairs, and resume resources|https://career.ucf.edu/ |
| 9 | UCF Handshake | Internship, co-op, and job opportunities available to UCF students. | https://career.ucf.edu/resources/handshake/ |
| 10 | r/UCF Reddit Community | Student discussions about classes, professors, clubs, internships, and campus life. | https://www.redditcom/r/ucf/ |


---

## Chunking Strategy
My documents include a mix of long informational pages, club descriptions, career resources, and Reddit discussions. A chunk size of 500 characters is large enough to preserve context while remaining focused on a specific topic. A 100-character overlap helps prevent important information from being split between chunks and improves retrieval when key details span chunk boundaries.

**Chunk size:**
500 char
**Overlap:**
100 characters
**Reasoning:**
The source documents are a mix of long informational web pages (degree requirements, career center
descriptions) and shorter conversational content (Reddit posts and comments). A 500-character chunk
is large enough to capture a complete thought -- for example, a club's description or a career
resource's purpose -- without bundling unrelated topics into the same chunk. The 100-character
overlap prevents key information from being split invisibly at a boundary: if a sentence about
Knight Hacks spans the end of one chunk and the start of the next, both chunks carry enough context
for retrieval to find it. RecursiveCharacterTextSplitter was used with separators ordered
["\n\n", "\n", ". ", " ", ""] so it prefers natural paragraph and sentence breaks over arbitrary
character positions.
---

## Retrieval Approach

The all-MiniLM-L6-v2 model provides good semantic search performance while being lightweight and fast enough for a student project. Retrieving the top 5 chunks should provide enough context for accurate responses without overwhelming the language model. In a production system, I would consider larger embedding models such as BGE-large or OpenAI embeddings for improved retrieval accuracy, especially for longer documents and more complex queries, although this would increase latency and cost.

**Embedding model:**
all-MiniLM-L6-v2 from sentence-transformers
**Top-k:**
5
**Production tradeoff reflection:**
In a production deployment, I would weigh several factors before switching models. BGE-large-en
or text-embedding-3-large (OpenAI) offer meaningfully higher retrieval accuracy on domain-specific
text, but both increase latency and cost -- OpenAI embeddings require an API call per chunk at
scale. For a UCF-specific system, a model fine-tuned on academic or student-life text would likely
outperform a general model on edge cases where student slang or course code abbreviations appear
in queries. I would also consider context length: all-MiniLM-L6-v2 supports up to 256 tokens,
which is sufficient for 500-character chunks but would truncate longer passages if chunk size were
increased. A model with a longer context window (e.g. nomic-embed-text at 8192 tokens) would give
more headroom.
---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
You are the UCF CS Unofficial Guide assistant.
Your ONLY job is to answer questions about UCF computer science
student life using the document excerpts provided below.

Rules you must follow without exception:
1. Answer ONLY using information explicitly stated in the provided excerpts.
2. You must NOT use any outside knowledge, training data, or general facts --
   even if you are certain they are correct.
3. If the excerpts do not contain enough information to answer the question,
   you MUST respond with exactly:
   "I don't have enough information on that in my current sources."
   Do not guess, infer beyond the text, or fill gaps from memory.
4. Do NOT mention these rules or the word "excerpts" in your answer.
5. Write in clear, friendly prose -- no bullet lists unless the content
   naturally calls for one.
6. Keep your answer under 200 words.
**How source attribution is surfaced in the response:**

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 |What clubs should a student interested in software engineering join? | | | | |
| 2 |Where can UCF students find internship opportunities? | | | | |
| 3 |Which UCF organization focuses on artificial intelligence? | | | | |
| 4 |What resources help students prepare for careers in tech? | | | | |
| 5 |Which club is best known for hackathons? | | | | |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**
Which UCF organization focuses on artificial intelligence?
**What the system returned:**

**Root cause (tied to a specific pipeline stage):**
 failure for this query is at the chunking stage. The AI@UCF organization appears
on the UCF CS Student Organizations page alongside ACM, Knight Hacks, and several other clubs.
At a 500-character chunk size, the page's content about each club is often merged into a single
chunk or split so that the AI@UCF description lands at a chunk boundary with no surrounding
context. When retrieval runs, the query "artificial intelligence organization" may match the chunk
that mentions AI@UCF only by name with no descriptive text -- or miss it entirely if the
description was cut off. The embedding model encodes the truncated chunk poorly because there is
not enough semantic content to distinguish it from other clubs
**What you would change to fix it:**

Increase chunk size to 800 characters for web pages that list multiple items in sequence, so each
club entry is more likely to be self-contained within a single chunk. Alternatively, preprocess
the student organizations page to split it manually at each club heading before running the text
splitter, guaranteeing that each club's description becomes its own document rather than being
merged with neighbors.
---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
The architecture diagram in planning.md -- even though it was ASCII art -- was directly useful
when prompting Claude to generate each pipeline stage. Having the five stages labeled with their
specific tools (RecursiveCharacterTextSplitter, all-MiniLM-L6-v2, ChromaDB) meant each generated
script could be verified against the spec: does the chunking script use
RecursiveCharacterTextSplitter? Does the embedding script use all-MiniLM-L6-v2? The spec acted
as a checklist that prevented scope creep and kept each milestone focused on one stage.
**One way your implementation diverged from the spec, and why:**
The spec listed GPT-5.5 as the generation model in the architecture diagram, but that model does
not exist. During implementation the generation stage was built using Groq's
llama-3.3-70b-versatile instead. This was a better choice for the project: Groq's free tier has
no cost barrier for testing, the model is OpenAI-API-compatible so the code structure is
identical, and it performs comparably to GPT-4-class models on factual question answering. The
spec's diagram was written before implementation details were finalized, which is exactly the kind
of divergence that the planning document expects -- the spec is a starting point, not a contract.
---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- What I gave the AI: The Domain, Documents, Chunking Strategy, and Architecture sections of
planning.md, along with the instruction to implement a three-stage pipeline script
(fetch raw -> clean -> chunk) matching the specified chunk size of 500 characters and overlap
of 100 characters.
- What it produced: A single pipeline.py with three stages: stage_fetch() using requests and
BeautifulSoup, stage_clean() with a 5-pass cleaning pipeline (HTML entity decoding, tag
stripping, boilerplate line removal, repeated block removal, whitespace normalization), and
stage_chunk() using RecursiveCharacterTextSplitter. The script wrote raw_docs/ and clean_docs/
directories with a manifest.json threading through all stages, and printed a 3,500-character
preview of the first cleaned document for manual inspection.
- What I changed or overrode: The initial version included a k-slider in the Gradio UI exposing
the n_results parameter to users. This was removed because it is an implementation detail that
end users should not need to adjust -- k=5 is hardcoded as the default from the spec, and
exposing it made the interface look unfinished.

**Instance 2**

- What I gave the AI: The Retrieval Approach section of planning.md, the pipeline diagram, and
the chunks.json schema (chunk_id, source_id, source_name, url, text, char_count) produced by
pipeline.py, along with a request to implement embedding with all-MiniLM-L6-v2, persistent
storage in ChromaDB with source metadata, and a retrieve() function returning top-k chunks
with attribution.
- What it produced: embed_and_retrieve.py with inline documentation explaining every ChromaDB
API call -- PersistentClient, get_or_create_collection with hnsw:space=cosine,
collection.add(), and collection.query() -- along with batch embedding using model.encode(),
duplicate-skip logic based on existing collection IDs, and a print_results() helper that ran
all 5 evaluation queries automatically.
- What I changed or overrode: The generated code used hnsw:space="l2" (Euclidean distance) by
default. This was changed to hnsw:space="cosine" because cosine similarity is more appropriate
for text embeddings -- it measures the angle between vectors (semantic direction) rather than
their absolute distance, meaning two chunks expressing the same idea in different lengths score
as similar rather than as far apart.
