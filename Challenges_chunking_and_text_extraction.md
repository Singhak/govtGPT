# govtGPT challenge log: problems faced at every step and how we solved them

Project: RAG over IT Act 2000, IT Rules 2021, DPDP Act 2023, DPDP Rules 2025.
This file is updated after **every step**. Each step has its own challenges + fixes.

| Step | What | Where in this file |
|---|---|---|
| 1 | LLM basics, setup, API | Section F |
| 2 | Text extraction + chunking | Sections A, B, C, D |
| 3 | Embeddings + Qdrant | Section E (last table) |
| 4 | BM25 + hybrid search (RRF) | Section E, Section G |
| 5 | Metadata filters + reranking | Section H |
| 6 | Retrieval evaluation (gold set, Hit@k, MRR) | Section I |
| 7 | Hard questions, out-of-scope questions, dense vs hybrid | Section J |
| 8 | Grounded answer generation, citations, refusal | Section K |
| 8b | Faithfulness: real quote + wrong answer | Section L |
| 9 | Tool-calling agent (ReAct loop, loop guard, tool failures) | Section M |
| 10+ | LangGraph, production | added as we go |

Code: `govtgpt/ingest.py`, `govtgpt/store.py`, `govtgpt/search.py`, and the `stepNN_*.py` scripts.

## 30-second summary (say this first in an interview)

"Government PDFs are not clean text. We hit bilingual pages with broken Hindi
font encoding, running headers, footnotes, a table of contents that looks like
real headings, amended sections with a `3[43A.` prefix, and repealed or
struck-down sections still printed in the text. I solved these with per-document
configs, language filtering, a monotonic section-number rule, and metadata for
legal status. I verified the result by reading real chunks, which is how I caught
that Sections 43A and 66A were being missed."

---

## A. Text extraction challenges

| # | Challenge | What we saw | Root cause | Fix |
|---|---|---|---|---|
| 1 | **Garbled Hindi text** | DPDP Rules 2025: "आवेदक" came out as "आवेिक"; unreadable | PDF uses a legacy font with a wrong encoding map | Detect language per page (Hindi chars > Latin chars) and drop Hindi pages; keep the English half (pages 24-41) |
| 2 | **Bilingual pages / mixed lines** | IT Rules 2021 header "[भाग II—खण्ड 3(i)] भारत का राजपत्र" mixed into English pages | Gazette prints a Hindi header on English pages | Drop any line containing Devanagari characters |
| 3 | **Running headers on every page** | "SEC. 1] THE GAZETTE OF INDIA EXTRAORDINARY 3" inside the text | Page header, not content | Regex noise filter (gazette header, lone page numbers, `REGD. No`, `GI/2025`) |
| 4 | **Footnotes mixed into body** | IT Act: "3. Ins. by Act 10 of 2009 ..." looked like "Section 3" | Amendment footnotes sit at page bottom | Footnote regex (`Ins.`, `Subs.`, `Omitted`, `5* * *`) removes them |
| 5 | **Table of Contents looks like headings** | "43A. Compensation for failure..." appears in the ToC too, creating duplicate / empty sections | ToC lists the same section titles | Body headings require a dash after the title (`.—` / `.–`) for IT Act; skip text before Section 1 |
| 6 | **Amended sections not detected** | Sections 43A and 66A were missing from the first run (3A, 6A, 40A too) | Amended sections start as `3[43A. ...` (footnote number plus `[`) | Heading regex allows an optional `\d+\[` prefix |
| 7 | **Schedule tables look like sections** | Table items "1.", "2.", "3." inside Schedules matched as sections | Same numbering style | Stop section detection once a SCHEDULE heading is seen; treat each Schedule as its own unit |
| 8 | **Footnote numbers collide with section numbers** | Footnote "3." vs Section 3 | Both are small numbers at line start | **Monotonic rule**: a heading is accepted only if its number moves forward (3 -> 3A -> 4), with a gap of up to 5 for omitted sections |
| 9 | **Section titles missing** | DPDP Act shows "Section 8" with no title | Titles are *marginal notes* placed beside the text and extracted as stray lines ("Application / of Act.") | Left empty and documented; the margin text sometimes sticks to the end of a section |
| 10 | **Every document has a different layout** | One regex pulled 14 headings from DPDP Act, 299 from IT Act | Different drafting styles (Act vs Rules, with/without dash, with/without title) | Per-document `DocConfig` instead of one global parser |
| 11 | **Console / encoding issues on Windows** | `UnicodeEncodeError` (cp1252), bullet "•" turned into "â€¢" | Windows default code page and PowerShell `Set-Content` | `sys.stdout.reconfigure(encoding="utf-8")`, write files as UTF-8 |
| 13 | **Replacement characters and odd titles in the DPDP Act** (found in Step 7) | Text shows "it shall�", "collected��"; the title of Section 3 became "Subject to the provisions of this Act, it shall" | Dashes in the PDF font are not decoded (they become U+FFFD); the title rule takes the first short sentence when margin titles are missing | **Not fixed yet.** Plan: replace U+FFFD with a dash, and drop auto-titles for the DPDP Act |
| 14 | **Line breaks inside phrases** (found in Step 7) | "territory of\nIndia" breaks exact phrase matching in my gold check | PDF line wrapping | Normalise whitespace before matching (in checks; chunks keep the raw lines for now) |
| 12 | **Broken spacing inside words** (found in Step 5) | "twenty -four", "exp oses", "wit hin", "anddispose" inside IT Rules chunks | PDF text layer places letters with odd gaps; the extractor guesses spaces wrongly | **Not fixed yet.** Plan: clean-up pass (join " -" patterns, dictionary-based word repair). Impact: BM25 cannot match "twenty-four" to "twenty -four" |

## B. Chunking challenges

| # | Challenge | Why it matters | Decision |
|---|---|---|---|
| 1 | **Chunk size trade-off** | Too small: a clause is separated from its proviso/exception, meaning is lost. Too large: the embedding blends several ideas, retrieval gets noisy, cost goes up | 1800 characters (about 400 tokens); split on structure first, only long sections are split further |
| 2 | **Overlap** | Without overlap a sentence can be cut in half across two chunks and only one half is retrieved | 200-character overlap |
| 3 | **Where to cut** | Cutting at arbitrary positions breaks sub-sections `(1)`, `(a)` | Separators in order: `\n(` (clause start), newline, sentence end, space |
| 4 | **A chunk loses its identity** | A middle chunk has no heading, so the LLM cannot tell which Act/section it belongs to | Metadata on every chunk (`act`, `section`, `title`, `page`, `chunk_index/total`, `status`) **and** a contextual header line inside the text: `[IT Act 2000 \| Section 66A: ...]` |
| 5 | **Legal status is not in the text** | Section 66A is still printed in the IT Act but was struck down (Shreya Singhal, 2015). Section 43A is to be omitted by DPDP Act s.44(2) (commencement date must be checked) | `status` and `status_note` metadata, shown in the chunk header so the answer can warn the user |
| 6 | **Verification** | The parser "worked" but silently missed key sections | Print the units found per document and spot-check known sections (43A, 66A, DPDP s.8, IT Rules r.4, DPDP Rules r.3) |

## C. Source and data-quality challenges

| # | Challenge | Detail |
|---|---|---|
| 1 | **Broken or outdated links** | The MeitY IT Act link returned 404; one police-department copy was the original 2000 text without the 2008 amendments (no Sections 43A or 66A) |
| 2 | **Which copy is "current"?** | India Code copy chosen (36 pages, includes 2008 amendments); its page also says "under updation" |
| 3 | **Amendments not included** | IT Rules 2021 file is the original Feb-2021 notification; later amendments (e.g. Oct 2022) are not in it. Recorded in metadata note |
| 4 | **Phased commencement** | DPDP Rules 2025 come into force in stages (some rules immediately, Rule 4 in Nov 2026). Same Act, different effective dates |
| 5 | **No official Digital India Bill text** | The Digital India Act has no published official draft (only 2023 consultation material; reported stalled in Dec 2025). Not ingested as law; if added later it must carry `status: draft` |

## D. Known limitations (be honest about these)

- DPDP Act section titles are not extracted.
- IT Act: now 119 sections detected (was 109). The 10 missing ones, including 67A, 67B, 69 and 69A, were found by the Step 6 gold-set check; see Section I. Some other wrapped-title sections could still be missed.
- Section numbers are accepted only if increasing, so a section after a large gap of omitted sections could be skipped.
- Margin notes can stick to the end of a section in DPDP Act.
- Only English text is indexed; Hindi versions are not.

## E. Retrieval challenges: issues with normal (dense) vector search

Dense search turns text into vectors and finds chunks by **similarity of meaning**.
Setup: `gemini-embedding-2` (3072 dims), Qdrant local mode, 290 chunks.

| # | Issue | What happens | What we saw in our project |
|---|---|---|---|
| 1 | **Weak on exact terms and numbers** | "43A" and "43" get almost the same embedding | "What is Section 43A?" also returned IT Act s.43 and DPDP s.43 |
| 2 | **Rare words and abbreviations** | Model has seen them less, so they are poorly represented | Legal short forms, case names (e.g. Shreya Singhal) |
| 3 | **Small words that change legal meaning** | "shall" vs "may" almost vanish in the embedding | Obligation vs permission |
| 4 | **Results are hard to explain** | A vector cannot show *why* a chunk matched | 0.72 vs 0.74: no visible reason |
| 5 | **Scores are poorly separated** | All scores sit around 0.7-0.8, so a "correct" and a "roughly correct" chunk look alike | No safe score threshold |
| 6 | **Domain shift** | General-purpose model is weaker on legal drafting ("notwithstanding", "save as otherwise provided") | Intimate-images question returned only Rule 3 chunks; right clause not guaranteed |
| 7 | **Long chunk dilutes the signal** | One vector is an average of many ideas; one key sentence gets buried | Large Rule 3 chunks |
| 8 | **Sensitive to query wording** | "arrest" vs "punishment" vs "jail" give different results | Paraphrase instability |
| 9 | **Neighbouring chunks crowd the top-k** | Several chunks of one section fill all slots | Four Rule 3 chunks for one question |

### Why not keyword (BM25) alone?
BM25 matches exact words, so it is perfect for "43A" but blind to synonyms.
Users say "can police catch me?" while the law says "arrest" and "offence".

> **Correction after Step 7 (Section J):** the argument below is the *textbook* case for
> hybrid search. On our own data, measurement did **not** support it: dense search alone
> won on exact lookups and on everyday-language questions. Read Section J before quoting
> this in an interview.

### Why hybrid search (the textbook argument)
The two methods fail in opposite places, so combining them covers both:

| | Dense (vector) | Sparse (BM25) |
|---|---|---|
| Strong at | meaning, paraphrase, everyday language | exact words, section numbers, names |
| Weak at | numbers, rare terms | synonyms, language mismatch |

Method: run both, take each top list, merge with **Reciprocal Rank Fusion (RRF)**.
RRF adds up **ranks, not scores**, so the two different score scales do not need to be
compared. Legal questions come in both types ("What is Section 66A?" and "What is the
punishment for abusing someone online?"), so one system must handle both.

### Other practical problems met while embedding
| Problem | Fix |
|---|---|
| Gemini free-tier rate limit hit (`GoogleAPIError`) while embedding in batches | Batches of 50 with retry and increasing wait |
| Changing the embedding model later makes old vectors useless | Rebuild the whole index (build script deletes and recreates the collection) |
| Windows: harmless Qdrant `portalocker` message at script exit | Data is saved correctly; message is only cleanup noise |

## F. Step 1 challenges: setup and calling the LLM

| # | Challenge | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | **Model not available** | `gemini-2.5-flash` returned 404 | Model not offered to new API users | Switched to a currently available model; model name kept in `.env` (`GEMINI_CHAT_MODEL`) so it can change without code edits |
| 2 | **Server overload** | 503 "high demand" on one model | Provider-side capacity | Retry (`max_retries=2`) and switch model |
| 3 | **Free-tier daily quota** | 429 after about 20 requests/day on the larger model | Free tier limit per model | Moved to the lite model; fewer retries so a failing call stops fast instead of burning quota |
| 4 | **`.env` formatting** | "API key not valid"; no output from the script | Unbalanced quote and the model line glued onto the key line (key length 74 instead of 39) | One `KEY=value` per line, no quotes; checked key **length only**, never printed the key |
| 5 | **Two env vars for one key** | Warning "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" | System variable plus `.env` | Library prefers `GOOGLE_API_KEY`; warning is harmless |
| 6 | **Response `.content` is a list** | `'list' object has no attribute 'strip'` | Gemini 3 returns content blocks, not a string | Use `response.text` |
| 7 | **Hallucination even at temperature 0** | Model invented "exact" wording of IT Rules Rule 4(1)(a) | Temperature controls randomness, not truth | This is why we need RAG: answer only from retrieved text, with citations |

## G. Step 4 challenges: BM25 and hybrid search

| # | Challenge | Symptom | Root cause | Fix / decision |
|---|---|---|---|---|
| 1 | **Duplicate-key bug in RRF** | "SCHEDULE" chunks polluted hybrid results (e.g. for the Data Fiduciary question) | Every schedule has `section = "SCHEDULE"`, so different chunks got the same identity key and were merged | Added a hash of the chunk text to the key; reran and the fake entries disappeared |
| 2 | **BM25 ranked the wrong section first** | "What is Section 43A?" put DPDP s.42 above 43A | Stopwords ("what", "is", "section") not removed; BM25 only counts words | Hybrid still ranked 43A first because dense had it at 1 and BM25 at 2. **Open:** add stopword removal |
| 3 | **Tokenizing "43A" correctly** | A plain split could turn "43A" into "43" and "A" | Tokenizer choice | Regex `\w+` keeps "43a" as one token, distinct from "43" (the whole reason for BM25 here) |
| 4 | **Dense and BM25 scores cannot be added** | cosine about 0.7-0.8 vs BM25 0-30 | Different scales | Reciprocal Rank Fusion: uses ranks only, `1/(60+rank)` |
| 5 | **Dense still returns neighbours of 43A** | DPDP s.4 and s.43 in the dense list | Embeddings place "43" and "43A" close | Not removable in dense; hybrid keeps the right one on top |
| 6 | **Verification** | Easy to assume "hybrid is better" | One example proves nothing | Compared dense / BM25 / hybrid side by side on 4 questions; formal measurement waits for Step 6 |

## H. Step 5 challenges: metadata filters and reranking

| # | Challenge | Symptom | Root cause | Fix / decision |
|---|---|---|---|---|
| 1 | **Filter in two places** | Dense search (Qdrant) and BM25 (in memory) filter differently | Two separate engines | Qdrant filter on `metadata.*` for dense; same rule applied in Python before BM25 ranking |
| 2 | **Should struck-down law be hidden?** | Hiding 66A would make "Is 66A valid?" return nothing | Silent filtering hides the real answer | Filters are **opt-in from user intent**; by default show the section with its `status` label |
| 3 | **No dedicated reranker in the Gemini stack** | Cross-encoders need a separate model download | Stack choice | Used the chat model as an LLM judge: one structured-output call per query scoring 0-10 per candidate |
| 4 | **Reranker can make results worse** | Intimate-images question: correct chunk (Rule 3, chunk 5) was first after hybrid, but a Schedule chunk jumped above it after rerank | The reranker is another model that can be wrong; no ground truth yet | **Do not trust it blindly.** Measure hit rate with and without it in Step 6; keep only if it wins |
| 5 | **Cost and latency** | An LLM call per query | Reranker reads every candidate in full | Rerank only a small pool (top 10), truncate each chunk to 900 characters, use one batch call |
| 6 | **Hand-checking the "right answer"** | Needed to know whether the correct clause was even in the index | No gold set | Searched chunk text directly: the 24-hour clause is Rule 3(2)(b), stored in Rule 3 chunk 5. This is the seed of the Step 6 evaluation set |
| 7 | **Spacing damage hurts retrieval** | Clause is stored as "twenty -four hours" | Extraction issue (see A12) | Logged; clean-up pass planned |

## I. Step 6 challenges: retrieval evaluation

Gold set: 30 questions in `artifacts/eval_set.json`; each has the correct Section/Rule.
Metrics: **Hit@k** = share of questions whose correct section is in the top-k;
**MRR** = average of 1/rank of the first correct result (rewards ranking it high).

### Results (298 chunks)
| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| Dense | 0.87 | 1.00 | 1.00 | 1.00 | 0.933 |
| BM25 | 0.77 | 0.97 | 0.97 | 0.97 | 0.869 |
| **Hybrid (RRF)** | **0.90** | 1.00 | 1.00 | 1.00 | **0.950** |
| Hybrid pool 10 + LLM rerank | 0.83 | - | - | - | 0.911 |

Reading: hybrid is best, BM25 alone is weakest, and the LLM reranker made results **worse**
(Section 43, 8 and 7 moved down), so it is not used by default.

### Challenges and fixes
| # | Challenge | Symptom | Root cause | Fix / decision |
|---|---|---|---|---|
| 1 | **Gold answers must be verified, not remembered** | Risk of a wrong "correct" answer making the numbers meaningless | My memory of which section covers what can be wrong | Each question has a `must` pattern that has to appear in the gold section's real text; script checks all 30 (0 wrong) |
| 2 | **The check found missing sections** | Sections 67A, 67B, 69, 69A (also 66B, 69B, 70B) were not in the index at all | Long titles wrap over two lines, and the dash that confirms a heading is on the second line, so the heading was rejected | Heading check now also looks at the next 2 lines for `.–`. IT Act sections 109 -> 119, chunks 290 -> 298 |
| 3 | **Duplicate points in the vector DB** | 588 points stored for 298 chunks | Rebuild after retries left old points; ids were random | Stable id per chunk (uuid5 of Act + section + chunk index + text), so re-runs overwrite; asserted unique |
| 4 | **Embedding rate limit: 100 requests/minute (free tier)** | HTTP 429 during indexing and again in evaluation | Each text counts as one request; evaluation embedded every question twice | Batches of 20, longer waits between retries; query-embedding cache (memory + `artifacts/query_embeddings.json`) so each question is embedded once, ever |
| 5 | **Index had to be rebuilt after a parser fix** | Changed chunks mean old vectors are stale | Vectors are tied to the exact chunk text | Rebuild script deletes and recreates the collection |
| 6 | **Reranker hurt results** | Hit@1 0.90 -> 0.83 | Small LLM as judge is noisier than hybrid retrieval | Not used by default; try a stronger judge or a real cross-encoder and re-measure |
| 7 | **Misses are mostly "neighbours", not wrong answers** | Rank 2 cases: 67B behind 67A, Act s.10 behind DPDP Rule 13, s.16 behind Rule 15 | Sibling sections and Act/Rules on the same topic look alike | Acceptable for retrieval; the answer step must pick from several sources. Possible later: rule-to-section linking in metadata |
| 8 | **The gold set is too easy** | Dense already scores Hit@3 = 1.00 | I wrote the questions from section titles, and titles sit inside the chunk header | Be honest: 30 easy questions cannot separate good methods; 0.90 vs 0.87 is one question. Next: add harder questions (everyday wording, scenarios, exact numbers) |

## J. Step 7 challenges: hard questions and questions the documents cannot answer

Three question sets, same 298 chunks. **Exact** = 10 lookups like "What is Section 43A?";
**Easy** = 30 questions phrased like section titles; **Hard** = 16 everyday-language or
scenario questions (gold may be several valid sections). Format: Hit@1 / MRR.

| Method | Exact (10) | Easy (30) | Hard (16) |
|---|---|---|---|
| **Dense only** | **1.00 / 1.000** | 0.87 / 0.933 | **0.81 / 0.885** |
| Hybrid 1:1 (BM25 + dense, RRF) | 0.90 / 0.950 | **0.90 / 0.950** | 0.44 / 0.640 |
| Hybrid, BM25 weight 0.3 + stopwords | - | 0.90 / 0.950 | 0.50 / 0.700 |
| Routed (hybrid only if query has "Section N"/"Rule N") | 0.90 / 0.950 | 0.87 / 0.933 | 0.81 / 0.885 |

**Honest conclusion:** hybrid won only on the easy set, by one question. On everyday wording it
fell far behind dense, and on exact lookups dense was already perfect. Default retriever is now
**dense** (`HybridSearcher.retrieve`). `hybrid()` stays in the code for re-testing on a bigger set.
Caution: sets are small (10 / 30 / 16), so treat these as strong hints, not proof.

### Challenges and fixes
| # | Challenge | Symptom | Root cause | Fix / decision |
|---|---|---|---|---|
| 1 | **My earlier claim did not hold** | In Step 4 I said dense search needs BM25 for "Section 43A" | Even then dense had 43A at rank 1; only neighbours below it were off. The contextual header `[IT Act 2000 \| Section 43A: ...]` inside each chunk already lets the embedding see the identifier | Measured instead of assuming; corrected the notes. Lesson: a textbook reason is a hypothesis until the evaluation agrees |
| 2 | **BM25 fails on everyday language** | Hard set: BM25 Hit@1 0.25, MRR 0.331 | Users say "fake profile / cheat"; the law says "personation / dishonestly". BM25 matches words, not meaning | Dense is the main retriever. BM25 also suffers from missing stopwords and damaged text (Section A12, A13) |
| 3 | **RRF lets a weak list outvote a strong one** | Hybrid 1:1 MRR 0.640 vs dense 0.885 on hard set; even BM25 weight 0.3 only reached 0.700 | A chunk at dense rank 15 plus BM25 rank 1 scores higher than dense rank 1 alone, since fusion uses ranks not confidence | Dropped hybrid as default; tested weights, stopwords and routing; none beat dense. Weighted or learned fusion needs a larger evaluation set |
| 4 | **One gold answer is not always right** | "My ex posted my private pictures": 66E, 67 and IT Rules 3 are all valid | Real legal questions touch several provisions | Gold is a **list** of acceptable sections; rank = best match |
| 5 | **Gold verification failed once** | Check for DPDP Act s.3 reported "unverified" | Text has "territory of\nIndia" (line break) and my pattern had a space | Verified by reading the chunk: gold was right, the check was too strict. New extraction notes A13 and A14 |
| 6 | **Definitions are hard to retrieve** | "What does data fiduciary mean?" put DPDP s.2 at rank 14 in hybrid | Definitions sit in one long section next to many chunks that use the term | Candidate for a later fix: boost or index definitions separately |
| 7 | **The bot must know when it cannot answer** | Out-of-scope questions (GDPR, Digital India Bill, IPC murder, CCPA, a recipe, Aadhaar) still return 2-3 legal-looking sections | Search always returns its top-k, even for nonsense | Measured whether a score can reject them: see below |

### Can a similarity score detect "not in the documents"?
Top-1 dense cosine: in-scope min 0.656, median 0.783; out-of-scope max 0.715, median 0.681.

| Threshold | Real questions wrongly rejected (of 46) | Out-of-scope rejected (of 7) |
|---|---|---|
| 0.66 | 1 | 3 |
| 0.70 | 6 | 5 |
| 0.72 | 9 | 7 |

No clean threshold: rejecting all 7 also rejects about 20% of valid questions. A score alone is
not enough. Plan: the answer step must check "does the retrieved text actually answer the
question?" and say "not found in the provided law" (Step 8), and keep a refusal test set.
Similar-sounding out-of-scope questions ("Digital India Bill", "GDPR right to be forgotten")
scored 0.68-0.69 and retrieved plausible DPDP sections, which is the dangerous case.

## Likely interview questions and short answers

**Q: What was the hardest part of ingestion?**
A: Government PDFs are layout-noisy: bilingual pages, broken Hindi encoding, footnotes, ToC and schedule tables that all look like section headings. I fixed it with per-document configs and a monotonic section-number rule.

**Q: How did you know your chunking was correct?**
A: I did not trust the parser; I printed units per document and read the chunks for known sections. That is how I found that amended sections (`3[43A.`) were missed.

**Q: Why did you add metadata and a header inside the chunk?**
A: A split chunk loses its heading. Metadata enables citation, filtering and status warnings; the header inside the text makes the embedding and the LLM aware of the source.

**Q: How do you handle repealed or struck-down law?**
A: A `status` field per section (for example `struck_down` for 66A), displayed in the chunk and in the answer, so the bot never presents it as current law.

**Q: Why hybrid search instead of plain vector search?**
A: Dense search captures meaning but is weak on exact tokens such as section numbers: in our test "What is Section 43A?" also returned Section 43, because "43" and "43A" embed close together. BM25 is exact on keywords but blind to synonyms. Legal queries need both, so I run them in parallel and fuse the ranks with Reciprocal Rank Fusion.

**Q: How do you choose an embedding model?**
A: Retrieval quality on my own domain, language coverage, max input length, dimension, cost and latency, and hosting or privacy needs. Benchmarks like MTEB only shortlist; the final choice comes from a small evaluation set measuring hit rate at top-k. Changing the model later means re-embedding the whole corpus.

**Q: Did reranking improve your results?**
A: No. On a 30-question gold set, hybrid alone had Hit@1 0.90 and MRR 0.950. Adding an LLM reranker dropped them to 0.83 and 0.911. A reranker is another model that can be wrong, so I keep it only if measurement shows a gain.

**Q: Did hybrid search beat dense search in your tests?**
A: Only on an easy set, by one question. On everyday-language questions dense had MRR 0.885 against 0.640 for hybrid, and on exact "Section 43A" style lookups dense was perfect. My contextual header inside each chunk lets the embedding see section identifiers, so BM25 added little. I made dense the default and kept hybrid for re-testing on a larger set. I would not claim hybrid is better without measuring it on my own data.

**Q: How does your system handle questions that are not in the documents?**
A: A similarity score alone cannot do it: on my data no threshold rejected all out-of-scope questions without rejecting about 20% of valid ones. The answer step therefore has to verify that the retrieved text answers the question and otherwise say it is not found, and I test that with a set of out-of-scope questions.

**Q: How do you evaluate retrieval?**
A: I build a gold set of questions with the correct section, verified against the real text. Then I measure Hit@k (is the right section in the top k?) and MRR (how high is it ranked?) for dense, BM25 and hybrid on the same set. I also read the failures, and I state the limits of the set: mine is small and easy, so small differences are noise.

**Q: What did evaluation reveal that you had not noticed?**
A: That ten sections, including 67A, 67B, 69 and 69A, were missing from the index. Their wrapped headings were rejected by my parser. I only found it because I verified each gold answer against the indexed text.

**Q: Should you filter out repealed or struck-down sections from search?**
A: Not silently. If a user asks "Is Section 66A valid?" and it is filtered out, the bot gives a wrong "does not exist" answer. I keep it with a status label and let filters follow the user's intent.

**Q: Why chunk size 1800 characters with 200 overlap?**
A: A balance: large enough to keep a clause with its proviso, small enough that the embedding stays focused. Overlap protects sentences cut at a boundary.


## K. Step 8 challenges: grounded answer generation

**What we built:** `govtgpt/answer.py`. Retrieve top-6 (dense) -> Gemini answers ONLY from the numbered passages -> every claim cites (passage number + exact quote) -> **code verifies each quote** -> legal-status warnings are added from metadata, not by the LLM.

**Result (`step08_answer.py`):** in-scope 4/4 answered, out-of-scope 7/7 refused ("not found in the provided documents"). The 66A question was answered *with* a struck-down warning. Small test set, so this is a smoke test, not a benchmark.

| # | Challenge | Symptom | Root cause | How we overcame it |
|---|---|---|---|---|
| 1 | **A score threshold cannot say "not found"** | Step 7: threshold 0.72 rejected 7/7 out-of-scope but also 9/46 valid questions | Similarity measures closeness, not whether the answer exists | Let the LLM judge `answerable` from the passages, with the rule "a passage that merely mentions the topic is NOT an answer" |
| 2 | **LLM may cite wrongly or invent a source** | First run: model cited "DPDP Rules Fourth Schedule" that was never retrieved, and gave labels like "Rule 3: (1)" | Free-text citations are unreliable | Cite by **passage number** only. Act and section are read from metadata by our code. Citations pointing to missing passages are dropped |
| 3 | **A citation can exist but the quote can be made up** | Quote not actually in the passage | LLM paraphrases while "quoting" | Code checks every quote is inside the cited passage. If all citations fail, the answer is replaced by "not found" (never show an unsupported answer) |
| 4 | **Honest quotes were rejected (over-refusal)** | First verified run: only 1/4 valid questions answered | (a) PDF spacing damage ("wit hin", "totake") while the LLM quotes repaired words (A12); (b) "twenty-four" vs "twenty four"; (c) the model joined two fragments with "..." | Compare after removing everything except letters and digits; split a quote on "..." and require every fragment to be present. 1/4 -> 4/4 |
| 5 | **Verifier strictness is a trade-off** | Too strict = refuses valid questions; too loose = accepts invented quotes | Both errors are real | Kept the check strict on *content* (all letters/digits must match) but tolerant on *formatting*. Out-of-scope stayed 7/7 |
| 6 | **Struck-down law presented as current** | 66A text is still in the PDF, the model could describe the punishment as if enforceable | Model reads text, not legal history | Status warning is added in code from `status` metadata (struck_down, superseded, draft). The model is also told to mention it. Do not rely on the LLM alone for legal status |
| 7 | **Partial citation drops stay silent** | One of two citations dropped, answer still shown | Verification only removes bad citations | Warnings list shows "Dropped citation [n]". Open: for the 66A answer one quote still fails, the answer text is correct but less cited |

**How "not found" is decided (three paths)**
1. The LLM sets `answerable=false`.
2. The LLM answers but gives no citation.
3. The LLM cites, but our code finds the quotes are not in the passages, so every citation is dropped.

Path 3 exists because the LLM can wrongly say "answerable" too. The code is the last guard.

**Why a score cannot do this job (precise version):** similarity measures how close the *topic* of question and chunk are, not whether the chunk *contains the answer*. A chunk can be very close and still not answer (e.g. a neighbouring section). Measured: threshold 0.72 stopped 7/7 out-of-scope but also 9/46 valid questions, so no single number separates them.

**Open weakness to test next:** the verifier checks that a quote *exists* in a retrieved passage, not that it *answers* the question. If the LLM says `answerable=true` and quotes a real but irrelevant line, the verifier would accept it. A faithfulness check (second LLM judge, or an answer-level test set) is needed.

**Lesson from the debugging:** when the first verified run refused valid questions I first suspected the model. Printing the model output showed the answer was right and my matcher was wrong (PDF spacing, hyphens, "..."). Check the checker before blaming the model.

**Open problems after Step 8**
- Citations are verified for *existence*, not for *entailment* (a real quote could still not support the claim). A second LLM check or human review would be needed.
- Answer quality is not scored yet. We need a small answer-level test set (faithfulness, correct refusal) like Steps 6-7 did for retrieval.
- Spacing damage (A12) still makes quotes look broken to the user. A text clean-up pass is pending.
- The LLM can still say "answerable" for a related-but-not-answering passage. We measured only 7 out-of-scope questions.

**Interview Q&A**

*Q: How do you stop hallucination in RAG?*
A: Four layers: retrieve only relevant chunks, instruct the model to answer only from them, force citations (passage + exact quote) and verify the quotes in code, and refuse with "not found" when verification fails. I measured it: 7/7 out-of-scope refused, 4/4 in-scope answered.

*Q: Why can't you just use a similarity threshold to refuse?*
A: Retrieval always returns top-k. At a 0.72 cosine threshold I rejected all 7 out-of-scope questions but also 9 of 46 valid ones. Score tells closeness, not answerability, so the answer step must decide.

*Q: Why do you verify citations in code instead of trusting the LLM?*
A: The model produced a citation to a passage that was never retrieved. Code-side checks (passage exists, quote is inside it) cannot be argued with. Legal-status warnings are also added from metadata for the same reason.

*Q: What did strict quote checking break?*
A: It over-refused (1/4 valid answered) because of PDF spacing damage, hyphen differences and "..." in quotes. I normalised to letters and digits and checked each fragment. This shows a verifier must be tested on valid cases too, not only on attacks.


## L. Step 8b challenges: "real quote, wrong answer" (faithfulness)

**The weakness (found while explaining Step 8):** Step 8 only checks that a quote *exists* in the cited chunk. It never checks that the *answer follows from* the quote. A wrong answer with a real quote passes.

**What we built:** `verify()` in `govtgpt/answer.py` now has layers: L1 quote exists, **L2 every number in the answer must appear in a cited quote (code)**, **L3 optional LLM judge** reads answer + cited passages. Test: `step08b_faithfulness.py`.

**Experiment:** 12 questions -> 11 honest answers (1 correctly refused). We then made 36 tampered answers that keep the real citations: T1 numbers changed (9), T2 answer swapped from another question (11), T3 meaning flipped, no number change (10), **T4 a wrong figure that IS inside the cited quote (6)**. T4 was added after Anil pointed out that a number check only sees *presence*, not *which claim the number belongs to*.

| Check | T1 wrong number | T2 swapped answer | T3 flipped meaning | T4 wrong figure inside quote | False alarms on honest |
|---|---|---|---|---|---|
| L1 quote only (old Step 8) | 0/9 | 0/11 | 0/10 | 0/6 | 0/11 |
| L1 + L2 numbers (code) | **9/9** | 8/11 | 0/10 | **0/6** | 1/11 |
| L1 + L2 + L3 LLM judge | 9/9 | **11/11** | 9/10 | **6/6** | 1/11 |

So the old verifier caught **0 of 36** bad answers. T4 shows exactly why the numbers check cannot stand alone: the figure is in the quote, so it passes; only the judge notices the figure answers a different question. This confirms the gap was real.

| # | Challenge | Symptom | Root cause | How we overcame it |
|---|---|---|---|---|
| 1 | **Existing quote does not prove the answer** | Tampered answers passed with 0/30 caught | Step 8 verified existence, not support | Added L2 (numbers) and L3 (judge) |
| 2 | **Numbers are written differently** | Answer "250 crore" vs text "two hundred and fifty crore"; "twenty four" vs "24"; PDF gives "twenty -four" | Law text uses words, LLM answers use digits | `numbers_in()` converts words and digits (crore/lakh/thousand) to one integer, so both forms match |
| 3 | **Numbers alone miss meaning flips** | T3 (must -> need not) caught 0/10 by numbers | No number changed | LLM judge caught 9/10. Code cannot read meaning |
| 3b | **Right number, wrong claim (T4)** | A chunk holds both "24 hours" and "15 days"; answer gives the wrong one, quote contains both | Number check tests presence only, not which claim a number belongs to | Numbers check stays as cheap first layer; the judge is what catches it (6/6). Never rely on L2 alone |
| 4 | **Swapped answers without numbers** | T2: numbers layer caught only 8/11 | Some answers contain no figures | Judge catches the rest (11/11) |
| 5 | **False alarm on an honest answer** | Obscene-material question refused: answer mentioned "five lakh", but the cited quote did not contain it | The answer used a fact from the chunk that the model did not quote | Kept as is: the number really was uncited. Cost: 1/11 honest answers refused. Option: also allow numbers from the whole cited passage (weaker check) |
| 6 | **Judge costs one extra LLM call per answer** | Latency and quota | Judge runs per question | Judge is optional (`use_judge=True`). Numbers check is free, use it always |

**Honest limits of this experiment (say these in an interview)**
- Tampered answers were made by *us*, not by a real misbehaving model. T1 is easy by construction, so 9/9 is not a general guarantee. T4 answers are blunt ("The figure asked about is ..."), so the judge's 6/6 is also optimistic; a subtle real error would be harder.
- 11 honest answers, 30 tampered: small. Treat the percentages as indicative.
- The judge is the same model family as the generator, so they may share blind spots. A different model as judge is a worthwhile test.
- One flipped-meaning case (1/10) slipped through even the judge.
- Still no test for *over-refusal* on harder valid questions.

**Decision:** numbers check ON by default (free, caught all number changes); judge ON for high-stakes answers (penalties, deadlines), OFF for cheap/fast paths.

**Interview Q&A**

*Q: Your answers are cited. Is that enough to trust them?*
A: No. A citation proves the quote exists, not that the answer follows from it. I tested 30 tampered answers that kept real quotes: the old quote-only check caught 0. Adding a number check in code caught all 9 wrong-number cases, and an LLM judge on top caught 29 of 30 overall.

*Q: Why not just use an LLM judge for everything?*
A: It costs an extra call and can share the generator's blind spots. A code check on numbers and figures is free and exact, so I run it first and use the judge only where a mistake is expensive.

*Q: What are the limits of your evaluation?*
A: Small set, and I created the bad answers myself, so the numbers are indicative. Next I would use a different judge model and a real answer-level test set.


## M. Step 9 challenges: a tool-calling agent

**What we built:** `govtgpt/agent.py`, written as an explicit loop (no framework magic, so every rail is visible). Tools: `search_law(query, act)` and `get_section(act, section)`. The LLM picks tools, we run them, results go back, repeat until the LLM says DONE. The agent only **gathers evidence**; the final answer still goes through the Step 8 grounded generate + verify, so the agent cannot skip citation checks. Demo: `step09_agent.py`.

**Rails in code:** `MAX_STEPS=6` hard cap; identical repeat call (same tool + args) is blocked; tool exceptions become an `ERROR:` message the LLM can read; at most 12 evidence chunks reach the answer step.

**What we observed (5 demo runs, not a benchmark)**

| Run | What happened |
|---|---|
| 1 Exact lookup (66E) | Used `get_section` (right tool), but then repeated the same call (loop guard **blocked** it) and made 2 more searches it did not need. Answer correct |
| 2 Compare 43A vs DPDP | Two tools in the first step, then 5 more steps and hit MAX_STEPS. 13 chunks collected. **The plain one-shot RAG baseline also cited both IT Act 43A and DPDP s.44** |
| 3 Search fails twice | Both failures became ERROR messages; agent did not crash, but then **guessed section numbers** (33, 34, 25, 32) and ended with a vague answer ("penalty specified in the Schedule"), missing s.44 and the amount |
| 4 Section 999 (does not exist) | `get_section` returned a helpful ERROR; the agent stopped after 1 call. Loop guard was not needed |
| 5 Out of scope (CCPA) | 1 search, stopped, refused. Cheap and correct |

| # | Challenge | Symptom | Root cause | How we overcame it |
|---|---|---|---|---|
| 1 | **Agent repeats itself** | Run 1 repeated the identical `get_section` call | LLM forgets it already has the result, or hopes for a different answer | Code keeps a set of (tool + args); a repeat returns "BLOCKED: change query or reply DONE" without running the tool |
| 2 | **Agent never knows when to stop** | Run 2 used all 6 steps and 13 chunks | "Enough evidence" is a judgement, and the model keeps searching | Hard `MAX_STEPS` cap plus an evidence cap. Honest cost: when the cap hits we may stop early |
| 3 | **A tool fails** | Run 3: search backend returned 503 twice | Real APIs fail (rate limit, network) | Exceptions are caught and returned as `ERROR: ...` text so the LLM sees them instead of the program crashing |
| 4 | **Recovery can be poor** | After the failures the agent guessed sections at random | Model had no fallback plan, only the two tools | Open. Ideas: automatic retry with backoff inside the tool, a clearer system prompt on what to do after errors, a cheaper fallback tool |
| 5 | **Tool errors should teach** | Section 999 | A bare "not found" gives the LLM nothing to act on | `get_section` errors list valid acts / some valid section numbers, and the agent stopped sensibly |
| 6 | **Agent could bypass grounding** | If the agent wrote the final answer itself it could use memory and skip citations | Agent output is free text | Agent only collects evidence; Step 8 `generate` + `verify` writes the answer, same checks as before |
| 7 | **Do we even need an agent?** | Run 2: one-shot RAG already found both sections | The question was multi-part but the retriever handled it | **Not proven.** Agent adds cost (6 LLM calls vs 1) and variance. Use an agent only where one retrieval is not enough; we have no such failing example yet |

**Open problems**
- No agent-vs-RAG benchmark: need questions where one-shot RAG really fails (e.g. "list every penalty in the DPDP Schedule that exceeds 100 crore", or questions needing 2 hops).
- Cost and latency not measured (steps x LLM calls).
- No memory across questions (each run starts empty). State and memory come with LangGraph.
- Parallel tool calls happen in one step (run 2, step 1) but we execute them one after another.
- Evidence cap silently drops chunks beyond 12 (run 2 collected 13).

**Interview Q&A**

*Q: What is the difference between an agent and a chatbot or plain RAG?*
A: Plain RAG always does one fixed pipeline: retrieve then answer. An agent decides at run time which tool to call, how often, and when to stop. That flexibility costs more LLM calls and adds failure modes, so it must be justified.

*Q: How do you prevent infinite loops in an agent?*
A: In code, not in the prompt: a hard step cap, detection of identical repeat calls, and a final fallback answer when the cap is hit. I saw the repeat guard fire on a real run.

*Q: What happens when a tool fails?*
A: I catch the exception and return it to the model as an error message. The agent survived two backend failures. Its recovery was weak (it guessed sections), so the next improvements are retries inside the tool and clearer instructions after errors.

*Q: Did the agent beat plain RAG?*
A: On my 5 test runs, no. Plain RAG answered the comparison question too. I would only keep an agent for cases a single retrieval cannot solve, and I would measure that with a proper test set before claiming a benefit.

**Clarifications from the teaching Q&A (Step 9)**
- *Agent vs plain RAG (Anil's words, refined):* plain RAG is one fixed path, retrieve then generate. In an agent the **LLM is the brain, tools are the hands**: it looks at each tool result, decides the next tool or to stop, and loops. The LLM is not a "tool"; RAG retrieval can be one.
- *Why loop limits live in code:* a prompt is a request, code is a lock. The LLM can skip or overlook a prompt rule (it repeated the same call on a real run despite the prompt). We still ask for DONE in the prompt because it helps in normal cases, but the guarantee is the step cap + repeat block.
- *Cost of the lock:* a step cap can stop the agent half-way (run 2 hit the cap). Fallback: build the answer from the evidence collected so far, through the Step 8 verifier.

**Where we stopped (end of session 1)**
Done: Steps 1 to 9 (incl. 8b and T4). Next: **Step 10 LangGraph** (StateGraph, state and memory, conditional routing, checkpoints, human approval before a risky action, self-correction loop). Good first test: a question where one-shot RAG really fails, so we can show whether the agent/graph helps.

**Housekeeping still open:** add pypdf, langchain-text-splitters, qdrant-client, langchain-qdrant, rank-bm25, langgraph to `requirements.txt`; add `artifacts/`, `qdrant_db/`, `data_txt/` to `.gitignore`; fix PDF spacing damage (A12) and U+FFFD / titles in the DPDP Act (A13); build an answer-level test set and try a different model as judge.


---

## N. Embedding model comparison: Gemini vs open-source BGE

**Setup.** Same 298 chunks, same BM25, same Qdrant (cosine). Only the embedding model changed, selected by `EMBED_BACKEND` / `BGE_MODEL` in `govtgpt/store.py`. Each model has its own DB folder and query cache, so the indexes do not overwrite each other. BGE runs locally on CPU (no rate limits, no data leaves the machine).

| Set | Retriever | Gemini-embedding-2 (3072d) | bge-large-en-v1.5 (1024d) | bge-m3 (1024d) |
|---|---|---|---|---|
| Easy (30) | dense | Hit@1 .87, MRR .933 | Hit@1 .83, MRR .903 | Hit@1 .80, MRR .883 |
| Easy (30) | hybrid | Hit@1 .90, MRR .950 | Hit@1 .90, MRR .942 | Hit@1 .87, MRR .933 |
| Hard (16) | dense | Hit@1 .81, Hit@5 1.00, MRR .885 | Hit@1 .69, Hit@5 .88, MRR .748 | Hit@1 .56, Hit@5 .81, MRR .701 |
| Hard (16) | hybrid | Hit@1 .44, Hit@5 .88, MRR .640 | Hit@1 .25, Hit@5 .75, MRR .495 | Hit@1 .38, Hit@5 .88, MRR .517 |

**Result:** Gemini > bge-large > bge-m3, mostly on the hard (paraphrased, plain-language) questions. On easy questions all three put the gold section in the top 3.

**Challenges and findings**
- **N1. Small eval sets.** 16 hard questions means one question is about 6 points. Treat the easy-set gaps as noise; only the hard-set gap is a real signal.
- **N2. Hybrid is capped by dense.** BM25 scored the same in every run, so hybrid gets worse when dense gets worse. Dense stayed the best retriever for every model.
- **N3. Query prefix.** `bge-*-en-v1.5` needs "Represent this sentence for searching relevant passages: " on queries only; `bge-m3` needs no prefix. Forgetting this silently lowers quality.
- **N4. Unanswerable-question score.** Cosine scales differ per model (Gemini out-of-scope max 0.715, bge-large 0.676, bge-m3 0.620). In all three the in-scope minimum is below the out-of-scope maximum, so no model gives a clean "not in the documents" threshold. bge-m3 has the biggest gap between the medians, but it still overlaps. Needs a separate check (LLM or reranker), as in Step 7.
- **N5. Cost trade-off.** BGE is free, offline and private, which matters for government data. The price is lower accuracy on paraphrased questions here.
- **N6. Caveats.** Local scripts print harmless shutdown tracebacks (portalocker / QdrantClient `__del__`). One hard gold label (DPDP Act s.3) is unverified for every model. Reranking and fine-tuning BGE were not tried.

**Interview-ready answer.** *Q: Did you try an open-source embedding model?* A: Yes. I swapped Gemini for bge-large and bge-m3 behind one env switch and ran the same retrieval tests. Gemini was best, mainly on hard paraphrased questions. bge-large was second. I would use BGE if privacy or cost came first, ideally with a reranker or fine-tuning.


---

## O. Step 10: LangGraph (StateGraph, routing, memory, self-correction)

**What was built.** `govtgpt/graph.py`: nodes `plan -> retrieve -> answer(verify) -> finalize`, plus `rewrite -> retrieve` on refusal. `MemorySaver` checkpointer, `thread_id` per conversation. Test: `step10_graph.py`.

**Why LangGraph.** Step 9 hand-rolled the loop. LangGraph gives typed shared state (with reducers), conditional edges, checkpoints (memory, resume, human-approval pauses) and a framework recursion cap. It does not improve retrieval or answer quality; the Step 8/8b verifier stays the last gate.

| Test | One-shot RAG | Graph |
|---|---|---|
| Hard in-scope answered (16) | 13 | 13 |
| Answered and cites a gold section | 12 | 13 |
| Out-of-scope refused (7) | 7/7 | 7/7 |

**Challenges and findings**
- **O1. No clear win on accuracy.** Answers differed on 6 of 16 questions, in both directions; an earlier run flipped some of them. Gap of 1 question is noise.
- **O2. Run-to-run variance.** Even at temperature 0, `numbers` refusals changed between runs (same question answered in one run, refused in the next). Need repeated runs / averages before claiming anything.
- **O3. Self-correction never fired.** Retry is routed only for `refused_by in (llm, quote)`; observed failures were mostly `numbers`. A rewrite cannot fix a generation/verifier-side failure. Next: a separate route for `numbers` (regenerate with fewer chunks).
- **O4. Memory works.** Same thread: "the same thing under the older IT Act" resolved to s.43A (superseded). New thread: the vague follow-up got a verified, cited but off-topic answer (electronic records). The verifier checks support, not relevance to the question.
- **O5. Out-of-scope handled in `plan`.** 7/7 refused with zero retrieval calls (saves cost). Risk: a wrong `in_scope=false` would refuse a valid question; not seen in this test but untested at scale.
- **O6. Cost.** Graph = plan call + generate (+ rewrite) vs one generate. More latency and 429/503 hits during the run.
- **O7. Checkpoint serialization warning.** LangGraph warns about unregistered `Result`/`Citation` types; register via `allowed_msgpack_modules` or store plain dicts before production.
- **O8. Not built.** Human-approval interrupt (no risky action in a read-only Q&A), persistent checkpointer (SQLite/Postgres), API/logging.

**Interview-ready answer.** *Q: Why LangGraph?* A: I wanted explicit state, visible routing, and checkpointed memory instead of a hand-written loop. I measured it against one-shot RAG on 16 hard questions: same accuracy within noise, 7/7 out-of-scope refusals saved retrieval, and memory resolved a follow-up that a stateless call answered wrongly. The retry loop never fired because failures were verifier-side, so I would not claim it improves accuracy.


---

## P. Step 11: route `numbers` refusals to a regenerate-with-feedback node (code done, measurement NOT done)

**Built.** `fix` node in `govtgpt/graph.py` (`fix_numbers=True` by default, `MAX_FIXES=1`): when `refused_by == "numbers"`, regenerate from the same evidence, telling the model which numbers were rejected and to state only figures that appear in a quoted passage. `A.generate()` gained an optional `feedback` argument. The verifier still runs on the regenerated answer. `step11_fix_numbers.py` compares one-shot vs graph (fix off) vs graph (fix on) over 2 repeats of the 16 hard questions.

**Result so far: inconclusive.**
- **P1. Daily quota hit.** The run crashed on question 2 of repeat 1: Gemini free tier allows 500 requests/day per model, and Steps 8 to 11 used them up (retry delay about 1h47m). The only data point: the fix fired once (hacker/customer-data question) and the answer was still refused by `numbers`. One case proves nothing.
- **P2. My retry wrapper hid the cause.** `retry()` swallowed the exception and printed "gave up", so the quota error was only visible after rerunning one call. Fix: print the error and stop immediately on a daily-quota 429 instead of sleeping 25s five times.
- **P3. Cost awareness.** The comparison costs about 8 LLM calls per question per repeat (one-shot 2, graph off about 3, graph on about 3 to 4). Budget the daily quota before running repeated evals.

**To finish.** Rerun `step11_fix_numbers.py` after the quota resets (or set `GEMINI_CHAT_MODEL` to another model, noting it changes the baseline). Do not claim the fix helps until it shows a gain across repeats.

**P4. Result after quota reset (2 repeats x 16 hard questions = 32 question-runs).**

| | Answered | Answered and cites gold |
|---|---|---|
| one-shot | 22/32 | 21/32 |
| graph, fix OFF | 28/32 | 26/32 |
| graph, fix ON | 31/32 | 28/32 |

- The `fix` node fired **once** (DPDP "maximum fine" question) and the answer was **still refused** (`numbers`). So the fix did not cause the 28 -> 31 gap between OFF and ON: those two graphs are identical except for that one event. That gap is run-to-run noise (about 3 questions of pure variance), which also tells us the size of noise in this setup.
- Graph vs one-shot (22 vs 28/31) is a bigger gap than that noise, but Step 10 showed 13 vs 13 on the same questions, so I treat it as suggestive only. One-shot itself moved from 13/16 (81%) to 22/32 (69%) between runs. Many 503/429 retries happened during this run.
- The "maximum fine" question refuses by `numbers` repeatedly, even with feedback. Likely cause (not verified): the amount is written in words/crore in the Schedule while the cited quote lacks it, or the model keeps restating it. Next: print the rejected numbers and the cited quote for this one question.
- Conclusion: `fix` route = no demonstrated benefit (1 firing, 0 successes). Keep it (cheap, only on refusals) but do not claim it helps. Multi-query planning in the graph is the more likely source of the graph-vs-one-shot gain, but I did not isolate it.

**P5. Why the "maximum fine" question refuses on `numbers` (debugged: 6 generations, same evidence).**
- 5 of 6 generations were fine (one citation, "two hundred and fifty crore", no unsupported numbers). 1 of 6 listed the whole penalty table (250, 200, 150, 50 crore, 10,000 rupees) with five quotes and also wrote **15 and 28**, which are not in any quote. So the refusal is intermittent, not deterministic, and the amounts were fine.
- Likely cause (read from the output, not yet checked against the passage): 15 and 28 are section references the model added from the Schedule's third column; they exist in the cited passage but not in the short quotes the L2 check allows.
- This is a **false refusal by the numbers check** (checker too strict for section references), the same family as Step 8's over-refusal. It also explains why the `fix` node did not help: the feedback says "numbers must be in a quote" but the model's fix is to re-list the table with the same section numbers.
- Not changed yet. Loosening L2 (allow numbers that appear in the cited passage text next to a quoted amount, or ignore numbers preceded by "section/rule/s.") must be re-tested with the Step 8b tampered-answer set (T1 to T4) so it does not start accepting wrong numbers. That is the next verifier task.


---

## Q. Step 12: FastAPI service and request logging

**Built.** `govtgpt/api.py`: `POST /ask` (question 3 to 500 chars, `thread_id` letters/digits/_/- up to 64) and `GET /health`. The graph is built once in the lifespan hook and reused. One JSON line per request goes to `logs/requests.jsonl`: request_id, thread_id, question, standalone question, queries, retrieved sections, cited sections, answerable, refused_by, warnings, retries, fixes, latency_ms, error. Test: `step12_api_test.py` (in-process TestClient).

**Smoke test (5 calls).**
| Case | Result |
|---|---|
| in-scope (DPDP penalty) | 200, answerable, cites DPDP s.44, 5.3 s |
| out-of-scope (CCPA) | 200, `answerable=false`, `refused_by=out-of-scope`, no retrieval, 14.8 s |
| empty / 600 chars / `../etc` thread_id | 422 each (pydantic validation) |
| log file | 2 lines, one per model call path, fields as above |

**Challenges and findings**
- **Q1. Refusal is not an error.** "Not in the documents" returns 200 with `answerable=false` and a `refused_by` reason, so clients and dashboards can tell it from failures.
- **Q2. Model outages map to 503.** RESOURCE_EXHAUSTED / UNAVAILABLE become 503 with the request_id and are logged with `error`; other exceptions are 500. This path is coded but **not tested** (needs a forced failure).
- **Q3. Single worker.** Local Qdrant locks its folder, so run one uvicorn worker. More workers need Qdrant server mode.
- **Q4. Memory is in-process.** `MemorySaver` is lost on restart and grows without limit; a SQLite/Postgres checkpointer is needed for production. Anyone who knows a `thread_id` can read that conversation's context, so there is no auth either.
- **Q5. Latency.** The out-of-scope call took 14.8 s although it did no retrieval; likely model retries on a busy model inside `plan`, not verified. No timeout is set on the LLM call yet.
- **Q6. Duplicate citations.** The in-scope answer lists s.44 twice (two quotes, same section). Cosmetic; group by section in the response.
- **Q7. Not built.** Auth, rate limiting per client, caching of repeated questions, log rotation, a PII policy for the logged questions (questions are stored in clear text).
- **Housekeeping closed.** `requirements.txt` now lists pypdf, langchain-text-splitters, qdrant-client, langchain-qdrant, rank-bm25, langgraph, fastapi, uvicorn (unpinned, as before); `.gitignore` has logs/, artifacts/, qdrant_db/, data_txt/. Not added: langchain-huggingface (used only for the BGE comparison).


---

## R. Step 13: reference-aware numbers check (fixes the P5 false refusal)

**Change** (`govtgpt/answer.py`, `verify(..., ref_aware=True)`). Numbers that name a provision ("section 15", "sections 15 and 28", "rule 4(2)", "s. 66E") are split out of the answer by `split_refs`. They are allowed if the number appears anywhere in a **cited** passage (or in the question). Everything else (amounts, durations) must still be inside a cited quote, as before. Old behaviour: `ref_aware=False`.

**Test** (`step13_numbers_check.py`, offline, cached 8b drafts, no LLM calls). L2 only, no judge.

| Case | n | Old L2 refused | New L2 refused | Want |
|---|---|---|---|---|
| Honest drafts | 11 | 1 | 1 | 0 |
| R1: replica of the real "max fine" answer (amounts in quotes, refs 8/15/28) | 1 | 1 | **0** | 0 |
| T1 wrong number | 9 | 9 | 9 | all |
| T2 swapped answer | 11 | 8 | 8 | all |
| T4 wrong figure present in quote | 6 | 0 | 0 | all (needs judge) |
| T5 invented section reference (section 999) | 11 | 11 | 11 | all |
| T6 reference to a real but uncited passage | 11 | 11 | 11 | all |

**Findings**
- **R1.** The replica is accepted by the new rule and refused by the old one. 8, 15 and 28 all appear in the Schedule passage, which supports the P5 explanation (section references from the Schedule column).
- **R2. No loss of protection** on T1, T2, T4, T5, T6: same counts as before. T5/T6 were already caught by the old rule (it refused every number); the new rule keeps catching them because a reference must exist in a cited passage.
- **R3. The one honest refusal remains** ("five lakh" not in the cited quote, the same case as Step 8b). That one is a true uncited figure, so it stays refused.
- **R4. Caveats.** R1 is a replica I wrote, not the model's actual failing output (that run was not captured). One honest case is 1 sample. `rule 4(2)` yields refs {4, 2}; the "2" is checked against all digits in the cited passages, which is loose (a stray "2" in a passage is enough). T4 is still 0/6 without the LLM judge, unchanged. Not re-run against the live model; the intermittent refusal rate (1 in 6) should be re-measured with real generations.
- **R5. Tooling gotcha.** A patch script turned `\b` into a backspace character inside the regex, silently making it match nothing; found only by testing the function. Always unit-test a new regex on examples.


---

## S. Step 13b/13c: live re-measurement and refinement of the numbers check

**Live measurement** (`step13b_live_rate.py`, fresh generations, same drafts verified under old and new rule, L2 only, no judge). 10 generations of the "max fine" question + 2 each of 11 other questions = 32 drafts (30 answerable... 32 model-answerable minus 0; 2 were `llm` refusals).

| | n answerable | `numbers`-refused, old rule | `numbers`-refused, Step 13 rule | after refinement (13c) |
|---|---|---|---|---|
| "max fine" (10 gens) | 10 | 3 | 2 | **0** |
| other 11 questions (2 gens each) | 22 | 3 | 3 | **0** |

So the false-refusal rate was **3/10 (not 1/6)** on that question, and 6/32 overall. The Step 13 rule alone fixed only 1 of 6.

**What the remaining 5 refusals actually were (read from the saved drafts)**
- **S1. Passage markers.** The model writes `[2]` inside its answer; the checker read "2" as a quantity (2 of 10 "max fine" drafts). Fix: drop `[N]` where N is a valid passage number.
- **S2. PDF spacing damage in the quote.** "twohundred and fiftycrore rupees" (1 draft): the number parser could not read it. Fix: `repair_spacing` on the SOURCE text only ("twohundred" -> "two hundred", "fiftycrore" -> "fifty crore", "fiv e" -> "five"); a token is split only if it is made entirely of number words.
- **S3. "five lakh" (2 drafts) was my earlier mistake.** I wrote in Step 8b that this figure was truly missing from the cited quote. It is in the quote as "fiv e lakh". The answer was correct and the checker wrongly refused it. Correction added to the story (Chapter 10).

**After refinement** (`step13c_recheck.py`, saved drafts re-verified, no LLM calls): numbers-refusals 0/32. Two drafts are still refused for another reason (`llm`: the model itself said not answerable); not a numbers problem.

**Tamper suite re-run** (`step13_numbers_check.py`): HONEST 0/11 (was 1/11), R1 0/1, T1 9/9, T2 8/11, T4 1/6 (was 0/6), T5 11/11, T6 11/11. Nothing tampered got easier to pass.

**Looseness check.** Repair applied to all 298 chunks: only 2 chunks change the set of numbers they contain: IT Act s.44 gains 150000 and s.67 gains 500000 (both correct readings of damaged text); each also drops one bogus artefact number (100050, 100000) produced by the damaged text. So the allowed set barely grows.

**Caveats**
- **S4.** T4 improved from 0/6 to 1/6 and I have **not** explained why; any change is in the safe direction here but I should not rely on it. T4 still needs the LLM judge.
- **S5.** Stripping `[N]` could hide a lone quantity written as `[3]`-style in the answer. Accepted: markers are what the model uses to cite, and only N from 1 to the number of passages is stripped.
- **S6.** Still small samples: 32 live drafts, 12 questions, one model, and the same model family generated and is judged. The tamper cases are still ones I created.
- **S7.** Rule 4(2)-style references still check the sub-number "2" loosely against the cited passages (see R4).
- **S8.** The numbers check is now a verifier with 3 helper heuristics (refs, markers, spacing). Each fixed a real observed false refusal; any new one should be added only with a failing real draft and a tamper re-run.
