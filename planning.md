# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
This project focuses on creating an Unofficial Guide to the UCF Computer Science Student Experience. The guide helps students find information about courses, degree planning, clubs, hackathons, internships, career resources, and campus opportunities that are relevant to computer science majors. This knowledge is difficult to find because it is scattered across many sources, including UCF department websites, student organization pages, career services resources, and community discussions on Reddit and Discord. A RAG system can bring these sources together and provide personalized recommendations and answers in one place.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

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

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 |What clubs should a student interested in software engineering join? |Knight Hacks, ACM, and Google Developer Student Club should be recommended. |
| 2 |Where can UCF students find internship opportunities? |Handshake and the Dixon Career Development Center should be identified as primary resources. |
| 3 |Which UCF organization focuses on artificial intelligence? |AI@UCF should be recommended. |
| 4 |What resources help students prepare for careers in tech? |Career fairs, Handshake, resume reviews, workshops, and career advising should be mentioned. |
| 5 |Which club is best known for hackathons and software |Knight Hacks should be identified. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.Information may become outdated because student organizations, events, officers, and internship resources change frequently throughout the year.

2.Some documents contain overlapping information, which may cause the retrieval system to return redundant chunks rather than diverse sources.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

Document Ingestion
(UCF Websites, Clubs, Reddit)
            |
            v
Chunking
(RecursiveCharacterTextSplitter)
            |
            v
Embedding + Vector Store
(all-MiniLM-L6-v2 + ChromaDB)
            |
            v
Retrieval
(Similarity Search)
            |
            v
Generation
(GPT-5.5)
            |
            v
Final Response


## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

     I will use ChatGPT and GitHub Copilot to generate document ingestion scripts and chunking functions. I will provide my Domain, Documents, and Chunking Strategy sections and ask the AI to create code that loads web pages, extracts text, and splits documents into chunks using a 500-character chunk size with 100-character overlap. I will verify that the generated chunks match my specified strategy.

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
