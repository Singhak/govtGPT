# govtGPT

A retrieval-augmented question-answering system over Indian IT and data-protection law, built step by step as a learning project. It retrieves the relevant sections, writes a short answer, and then **verifies the answer in code** before showing it. When the documents do not contain an answer, it says so instead of guessing.

**Corpus (4 documents, 298 chunks):** IT Act 2000, IT Rules 2021, DPDP Act 2023, DPDP Rules 2025.

> **Not legal advice.** The system describes what the text says. It does not know about court rulings beyond the status flags in its metadata, nor about amendments missing from the PDFs (see [Known limitations](#known-limitations)).

**Stack:** Python 3.14, LangChain, LangGraph, Gemini API (chat and embeddings), Qdrant (local mode), BM25 (`rank-bm25`), FastAPI.

---

## Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Using the API](#using-the-api)
5. [Repository layout](#repository-layout)
6. [Design decisions and what was measured](#design-decisions-and-what-was-measured)
7. [Evaluation results](#evaluation-results)
8. [Verification layers](#verification-layers)
9. [Known limitations](#known-limitations)
10. [Roadmap](#roadmap)
11. [Further reading in this repo](#further-reading-in-this-repo)

---

## What it does

- **Answers with citations.** Every claim carries a passage number and a short exact quote. The act and section shown to the user come from chunk metadata, never from the model's own wording.
- **Checks the answer, not just the quote.** Code confirms each quote exists in the cited passage and that every number in the answer is supported. An optional LLM judge checks meaning.
- **Refuses when it should.** Out-of-scope questions (GDPR, IPC, recipes, current affairs) are refused, and so are answers that fail verification.
- **Knows legal status.** Section 66A (struck down in *Shreya Singhal*, 2015) and Section 43A (superseded by DPDP Act s.44(2)) carry a `status` field. A warning is added from metadata, not from the model.
- **Remembers the conversation.** Follow-ups like "and what did the older IT Act say about the same thing?" are resolved per `thread_id`.

## Architecture

```
                       +-----------------------------------------------+
 question + thread_id  |                LangGraph (graph.py)           |
 --------------------> |                                               |
                       |  plan ----(out of scope)------------+         |
                       |   |                                 |         |
                       |   v                                 |         |
                       |  retrieve  <-----------+            |         |
                       |   |                    | rewrite    |         |
                       |   v                    | (llm/quote |         |
                       |  answer = generate     |  refusal)  |         |
                       |     + verify (answer.py)            |         |
                       |   |        \                        |         |
                       |   |         \-- refused ------------+         |
                       |   |             'numbers' -> fix (regenerate  |
                       |   |              once with feedback) -> answer|
                       |   v                                           |
                       |  finalize -----------------------------> END  |
                       +-----------------------------------------------+
                                  |
              Qdrant (dense, cosine) + BM25 (kept, not default)
                                  |
              298 chunks, section-aware, with legal-status metadata
```

| Node | What it does |
|---|---|
| `plan` | One LLM call: is the question in scope, rewrite it so it stands alone (uses chat history), and produce 1 to 3 search queries (one per law or sub-question). Out-of-scope questions skip retrieval entirely. |
| `retrieve` | Dense search per query, merged and de-duplicated, capped at 12 chunks. |
| `answer` | `generate` (LLM, structured output: answerable, answer, citations) then `verify` (code, see below). |
| `rewrite` | Only when the refusal was `llm` or `quote` (retrieval-side miss): new queries with different wording. Max 2 retries. |
| `fix` | Only when the refusal was `numbers`: regenerate once from the same evidence, telling the model which numbers were rejected. |
| `finalize` | Return the result and append to the thread's history. |

`MemorySaver` checkpoints state per `thread_id`. Safety limits live in code (retry counters, `recursion_limit`), not in prompts.

## Quick start

**Requirements:** Python 3.10+ (developed on 3.14), a Gemini API key.

```powershell
git clone <your-repo-url> govtGPT
cd govtGPT
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

copy .env.example .env      # then put your key in .env
```

`.env`:

```
GOOGLE_API_KEY=your-key-here
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite     # model names change; check the Gemini docs
# optional: EMBED_BACKEND=bge  and  BGE_MODEL=BAAI/bge-large-en-v1.5   (see below)
```

Place the four PDFs in `data/` (`it_act_2000.pdf`, `it_rules_2021.pdf`, `dpdp_act_2023.pdf`, `dpdp_rules_2025.pdf`), then build the index **once**:

```powershell
.venv\Scripts\python -W ignore step03a_build_index.py
```

This parses and chunks the PDFs, embeds the 298 chunks and writes the local Qdrant folder (`qdrant_db/`). On the Gemini free tier, embedding is rate limited (about 100 requests per minute); the code batches and retries.

Ask a question from Python:

```python
from dotenv import load_dotenv; load_dotenv()
from govtgpt.graph import build_graph, ask
from govtgpt.search import HybridSearcher

g = build_graph(HybridSearcher())
st = ask(g, "What is the penalty for failing to protect personal data under the DPDP Act?", "demo")
r = st["result"]
print(r.answerable, r.answer, r.warnings)
```

Notes:

- **Free-tier quota.** Gemini's free tier allows about 500 chat requests per day per model. Repeated evaluations can exhaust it (this happened during development). A daily-quota 429 is not worth retrying.
- **Optional open-source embeddings.** `EMBED_BACKEND=bge` runs BGE locally on CPU. It needs `langchain-huggingface` and `sentence-transformers`, which are **not** in `requirements.txt`. You must rebuild the index with the other model, because embeddings from different models are not comparable.
- **Windows console.** Scripts call `sys.stdout.reconfigure(encoding="utf-8")` so Hindi/Unicode prints correctly. Local Qdrant prints harmless `portalocker` / `__del__` tracebacks at shutdown.

## Using the API

Start the service (**one worker only**: local Qdrant locks its folder):

```powershell
.venv\Scripts\python -m uvicorn govtgpt.api:app --port 8000
```

```powershell
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" `
  -d '{"question": "Can a Data Fiduciary process childrens data without parental consent?", "thread_id": "user-42"}'
```

Response:

```json
{
  "request_id": "c0b1811294de",
  "answerable": true,
  "answer": "...",
  "citations": [{"act": "DPDP Act 2023", "section": "9", "status": "enacted", "quote": "..."}],
  "warnings": [],
  "refused_by": "",
  "latency_ms": 5277
}
```

| Field | Meaning |
|---|---|
| `answerable` | `false` means "the documents do not contain an answer". It is a normal 200 response, not an error. |
| `refused_by` | `""`, `out-of-scope`, `llm` (model said not answerable), `quote` (no citation survived), `numbers`, or `judge`. |
| `warnings` | Dropped citations, rejected numbers, and legal-status notes (`status` is `enacted`, `struck_down`, `superseded` or `draft`). |

| Status | When |
|---|---|
| 200 | Answered or refused. |
| 422 | Validation: question must be 3 to 500 characters, `thread_id` only letters, digits, `_`, `-`, up to 64 characters. |
| 503 | Model temporarily unavailable (quota or overload). Coded, **not yet tested**. |
| 500 | Any other internal error. |

`GET /health` returns `{"status": "ok"}`.

**Request log.** Each request appends one JSON line to `logs/requests.jsonl`: `request_id`, `thread_id`, `question`, `standalone_question`, `queries`, `retrieved`, `cited`, `answerable`, `refused_by`, `warnings`, `retries`, `fixes`, `latency_ms`, `error`. Questions are stored in clear text.

The API was smoke-tested in-process with FastAPI's `TestClient` (`step12_api_test.py`). It has not been load tested or run behind a real server.

## Repository layout

```
govtgpt/
  ingest.py     PDF -> clean text -> section/rule-aware chunks with metadata
  store.py      embeddings (Gemini default, BGE optional) + Qdrant local index
  search.py     dense retrieval (default), BM25, hybrid (RRF), metadata filters, LLM reranker
  answer.py     generate (structured output) + verify (quote, numbers, optional judge)
  agent.py      Step 9 tool-calling agent (explicit loop, loop guard); evidence only
  graph.py      Step 10/11 LangGraph pipeline (plan/retrieve/answer/rewrite/fix/finalize)
  api.py        Step 12 FastAPI service + JSON request log
data/           the four source PDFs
data_txt/       extracted text (generated)
artifacts/      gold sets, cached drafts, query embedding caches, chunks.jsonl
step01..step13  one runnable script per learning step (each prints its own results)
Project_Story_govtGPT.md                     the build story, written for interview retelling (Hindi + English)
Challenges_chunking_and_text_extraction.md   every problem found and how it was fixed, with tables
Short interview-ready answers in English.md  short Q&A
```

| Script | Purpose |
|---|---|
| `step01_llm_basics.py` | LLM alone: tokens, temperature, hallucination |
| `step02a/b_*.py` | Inspect PDFs, chunking demo |
| `step03a_build_index.py` / `step03b_search.py` | Build the index / first semantic search (pass a question as argument) |
| `step04_hybrid_demo.py`, `step05_filter_rerank_demo.py` | BM25 + RRF; filters and LLM reranking |
| `step06_evaluate.py` | Retrieval metrics on the 30-question gold set |
| `step07_hard_and_unanswerable.py` | Hard (paraphrased), exact-lookup and out-of-scope sets |
| `step08_answer.py`, `step08b_faithfulness.py` | Grounded answers; tampered-answer test of the verifier |
| `step09_agent.py` | Tool-calling agent: planning, tool failure, loop guard |
| `step10_graph.py` | Graph vs one-shot RAG, plus the memory test |
| `step11_fix_numbers.py` | Does the `numbers` fix route help? |
| `step12_api_test.py` | API smoke test (about 4 LLM calls) |
| `step13_numbers_check.py`, `step13b_live_rate.py`, `step13c_recheck.py` | Reference-aware numbers check: offline tamper suite, live refusal rate, re-check of saved drafts |

Run any of them with `.venv\Scripts\python -W ignore <script>`. Many make LLM calls and count against your daily quota.

## Design decisions and what was measured

These were decided by measurement on this corpus, not by assumption. Several early beliefs turned out to be wrong, and the repo keeps those corrections.

| Decision | Reason (measured) |
|---|---|
| **Parser fixes before any model work** | Most serious bugs were in extraction: Hindi garbage, repeating headers, footnotes read as sections, the table of contents read as sections, wrapped titles that hid sections 67B, 69 and 69A. Fixed with a monotonic section-number rule, wrapped-title look-ahead and footnote filters. IT Act sections went from 109 to 119. |
| **Section-aware chunks with a contextual header** (`[IT Act 2000 \| Section 43A: ...]`) | A section is one legal idea. The header also lets the dense embedding see the section number, which is why dense search handled exact lookups perfectly. |
| **Dense retrieval is the default; hybrid is kept but off** | Hybrid (BM25 + dense via RRF) won on the easy set but lost badly on paraphrased questions (Hit@1 0.44 vs 0.81), because a weak BM25 list drags the fused ranking down. |
| **Gemini embeddings (3072d) over BGE** | Same data, same index, one switch: Gemini beat `bge-large-en-v1.5` and `bge-m3`, mainly on paraphrased questions. BGE stays as the private/offline option. |
| **No LLM reranker by default** | It lowered Hit@1 from 0.90 to 0.83 on the easy set. |
| **No similarity-score threshold for "not in the documents"** | At a 0.72 cutoff, 7/7 out-of-scope questions were rejected but 9 of 46 valid ones too. Score measures topic closeness, not whether an answer exists. A generate-and-verify step decides instead. |
| **Citations verified in code** | The model once cited a Schedule that was never retrieved. The model gives only a passage number and a quote; code checks the quote and fills in act and section. |
| **Legal status from metadata** | Without it the system would say 66A carries three years' imprisonment. |
| **Agent only gathers evidence** | Loop limits and repeat-call blocking are in code. In 5 test runs the agent did not beat one-shot RAG, so it is not on the main path. |
| **LangGraph for state, routing and memory, not for accuracy** | It gives typed state, visible conditional edges and checkpointed memory. It did not improve retrieval or answer quality by itself. |

## Evaluation results

Everything here is from small hand-built sets (30 easy, 16 hard, 10 exact-lookup, 7 out-of-scope) and a single model family. Treat gaps of one question (about 6 points on the hard set) as noise.

**Retrieval (gold section in top-k), Gemini embeddings**

| Set | dense (Hit@1 / MRR) | bm25 | hybrid |
|---|---|---|---|
| Easy (30) | .87 / .933 | .77 / .869 | .90 / .950 |
| Hard (16) | **.81 / .885** | .25 / .331 | .44 / .640 |
| Exact (10) | **1.00 / 1.000** | n/a | .90 / .950 |

**Answer generation.** In-scope 4/4 answered, out-of-scope 7/7 refused (a small test, not a benchmark).

**Does verification catch wrong answers? (Step 8b, answers I tampered with while keeping the real quotes)**

| Check | wrong number | swapped answer | flipped meaning | wrong figure present in quote |
|---|---|---|---|---|
| quote check only | 0/9 | 0/11 | 0/10 | 0/6 |
| + numbers check (code) | 9/9 | 8/11 | 0/10 | 0/6 |
| + LLM judge | 9/9 | 11/11 | 9/10 | 6/6 |

A quote that exists does not make an answer correct. Code catches numbers, not meaning; the judge catches meaning but costs an extra call, so it is used on high-stakes questions only. The tampered answers were generated by me, so these numbers are optimistic.

**Graph vs one-shot (16 hard questions x 2 repeats, answered)**

| one-shot | graph, fix off | graph, fix on |
|---|---|---|
| 22/32 | 28/32 | 31/32 |

The 28 vs 31 difference is run-to-run noise (the fix fired once and did not succeed), so the fix is not shown to help. An earlier single run showed 13 vs 13. The graph's lead over one-shot is suggestive, not proven.

**False refusals of the numbers check (live generations)**

| | before | after the Step 13 refinements |
|---|---|---|
| answerable drafts refused by `numbers` | 6/32 | 0/32 |

The tamper suite was re-run after the change and nothing tampered became easier to pass (see `Challenges_...md`, sections R and S).

## Verification layers

`verify()` in `govtgpt/answer.py`, in order:

1. **L1 quote.** Every citation's quote must appear in the cited passage. Matching keeps only letters and digits, and `...` splits a quote into fragments that are each checked. This tolerates the PDF's broken spacing. If no citation survives, the answer is refused (`quote`).
2. **L2 numbers.** Every quantity in the answer must be supported by a cited quote, its passage header, or the question. A small parser makes `250 crore` and `two hundred and fifty crore` the same integer. Refinements, each added after a real false refusal:
   - provision references ("section 15", "rule 4(2)") must appear somewhere in a **cited** passage;
   - the model's own `[2]` passage markers are ignored;
   - PDF spacing damage in the source text is repaired before comparing ("fiv e lakh" to "five lakh").
3. **L3 judge (optional).** A second LLM call decides whether the answer is fully supported by the cited passages.
4. **Status warnings** are added from metadata for struck-down, superseded and draft provisions.

## Known limitations

- **Spacing damage in extracted text** ("wit hin", "totake") remains. It makes quotes look broken. The numbers check works around it, but the real fix belongs in ingestion.
- **DPDP Act text** has `U+FFFD` characters and some wrong auto-generated titles.
- **Definitions** (for example Section 2) rank low in retrieval.
- **Missing amendments:** IT Rules 2021 has no later amendments. The Draft Digital India Bill is not in the corpus. DPDP Rules commencement is phased; the system does not model dates.
- **English only.** Hindi text in the Gazette PDFs is deliberately dropped.
- **Small evaluation sets**, one model family for generation and judging, and tamper cases written by the author. No answer-level test set (faithfulness and refusal rates on a larger sample) yet.
- **Refusal is not a calibrated probability.** The model can still produce a verified but off-topic answer, especially for vague follow-ups without a `thread_id` history. The verifier checks that an answer is supported, not that it addresses the question.
- **Self-correction is thin.** Rewrite and fix routes rarely fire; no demonstrated benefit yet.
- **Production gaps:** no authentication or per-client rate limit, so anyone who knows a `thread_id` can read that conversation's context; memory is in-process and lost on restart (`MemorySaver`); no caching of repeated questions; questions are logged in clear text; one worker only (local Qdrant); no timeout on LLM calls; the 503 path is untested; LangGraph warns that checkpointing `Result`/`Citation` objects will be blocked in a future version.
- **Free-tier limits** on Gemini (daily and per-minute) affect indexing and evaluation.

## Roadmap

1. Fix spacing damage and `U+FFFD` in ingestion, rebuild the index, re-run the retrieval evals.
2. Build an answer-level test set (faithfulness, correct refusal) and use a different model as judge.
3. Persistent checkpointer (SQLite or Postgres), response caching, authentication, rate limiting, log redaction.
4. Human-approval pause (`interrupt`) once the system takes an action with side effects (for example drafting a notice).
5. Qdrant server mode for multiple workers.

## Further reading in this repo

- `Project_Story_govtGPT.md`: the full story, chapter by chapter, including what was assumed, what was measured, and what turned out wrong.
- `Challenges_chunking_and_text_extraction.md`: the challenge log (sections A to S) with tables and interview-ready answers.
- `Short interview-ready answers in English.md`: short Q&A form.
