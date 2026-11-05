"""Step 8 / 8b: grounded answer generation with layered verification.

generate (LLM)  ->  verify (code):
  L1  every citation's quote must be inside the cited passage        (Step 8)
  L2  every NUMBER in the answer must come from a cited quote        (Step 8b, code)
  L3  optional LLM judge: is the answer supported by the passages?   (Step 8b, LLM)
Legal-status warnings always come from metadata, never from the LLM.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from pydantic import BaseModel, Field

NOT_FOUND = "The provided legal documents do not contain an answer to this question."


class Citation(BaseModel):
    passage: int = Field(description="the passage NUMBER in square brackets, e.g. 3")
    quote: str = Field(description="a SHORT exact copy-paste (max 25 words) from that passage supporting the claim")


class Draft(BaseModel):
    answerable: bool = Field(description="true ONLY if the passages directly answer the question")
    answer: str = Field(description="plain-English answer using only the passages; empty if not answerable")
    citations: list[Citation] = Field(description="one per claim; empty if not answerable")


SYSTEM = (
    "You are a careful assistant for Indian IT and data-protection law.\n"
    "Rules:\n"
    "1. Use ONLY the numbered passages below. Never use outside knowledge, even if you know the answer.\n"
    "2. If the passages do not directly answer the question, set answerable=false. "
    "A passage that merely mentions a related topic is NOT an answer.\n"
    "3. Cite every claim with the passage number and a short EXACT quote copied from that passage.\n"
    "4. If a passage header says status: struck_down or superseded, say so in the answer.\n"
    "5. Do not give legal advice; describe what the text says."
)


@dataclass
class Result:
    question: str
    answerable: bool
    answer: str
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[Document] = field(default_factory=list)
    refused_by: str = ""        # "", "llm", "quote", "numbers", "judge"


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    # Keep ONLY letters/digits. The PDF text has broken spacing ("wit hin", "anddispose offsuch")
    # and odd punctuation ("twenty four" vs "twenty-four") while the LLM quotes the cleaned-up
    # words, so a plain substring test would wrongly reject honest quotes.
    return re.sub(r"[\W_]+", "", s).lower()


_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_UNITS.update({w: 10 * i for i, w in enumerate(
    "x x twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "x"})
_SCALES = {"thousand": 10**3, "lakh": 10**5, "lakhs": 10**5, "million": 10**6,
           "crore": 10**7, "crores": 10**7, "billion": 10**9}


def numbers_in(text: str) -> set[int]:
    """All quantities in the text, as integers. '250 crore' and 'two hundred and fifty crore'
    both become 2_500_000_000; 'twenty four' and '24' both become 24. Lone 0/1 are ignored."""
    toks = re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower())
    out: set[int] = set()
    i = 0
    while i < len(toks):
        t = toks[i]
        if not (t[0].isdigit() or t in _UNITS or t in _SCALES):
            i += 1
            continue
        acc = cur = 0
        seen = False
        j = i
        while j < len(toks):
            t = toks[j]
            if t[0].isdigit():
                if cur:
                    break
                cur = int(float(t.replace(",", "")))
            elif t in _UNITS:
                cur += _UNITS[t]
            elif t == "hundred":
                cur = (cur or 1) * 100
            elif t in _SCALES:
                acc += (cur or 1) * _SCALES[t]
                cur = 0
            elif t == "and" and (cur or acc) and j + 1 < len(toks) and (
                    toks[j + 1] in _UNITS or toks[j + 1] == "hundred"):
                pass
            else:
                break
            seen = True
            j += 1
        if seen:
            out.add(acc + cur)
        i = max(j, i + 1)
    return {n for n in out if n > 1}


# "section 15", "sections 15 and 28", "rule 4(2)", "s. 66E": numbers that NAME a provision, not a quantity.
_REF = re.compile(
    r"\b(?:sub-?sections?|sections?|rules?|clauses?|articles?|ss?\.)\s*\(?\d+[A-Za-z]{0,2}\)?"
    r"(?:\s*\([^)]*\))*(?:\s*(?:,|and|or|to)\s*\d+[A-Za-z]{0,2}(?:\s*\([^)]*\))*)*", re.I)


def split_refs(answer: str) -> tuple[str, set[int]]:
    """(answer with provision references blanked out, the numbers those references used)."""
    refs: set[int] = set()

    def grab(m):
        refs.update(int(x) for x in re.findall(r"\d+", m.group(0)))
        return " "
    return _REF.sub(grab, answer), refs


_VOCAB = set(_UNITS) | set(_SCALES) | {"hundred"}


def _split_vocab(tok: str) -> list[str] | None:
    """'twohundred' -> ['two','hundred'], 'fiftycrore' -> ['fifty','crore']; None unless the WHOLE
    token is made of number words (so ordinary words are never split)."""
    best: dict[int, list[str] | None] = {0: []}
    for end in range(1, len(tok) + 1):
        best[end] = None
        for w in _VOCAB:
            st = end - len(w)
            if st >= 0 and tok[st:end] == w and best.get(st) is not None:
                best[end] = best[st] + [w]
                break
    return best[len(tok)]


def repair_spacing(text: str) -> str:
    """Undo the PDF spacing damage that hides numbers: 'fiv e lakh' -> 'five lakh',
    'twohundred and fiftycrore' -> 'two hundred and fifty crore'. Used on the SOURCE text only."""
    toks = re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower())
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if i + 1 < len(toks) and t[0].isalpha() and toks[i + 1][0].isalpha():
            m = t + toks[i + 1]
            if m in _VOCAB and not (t in _VOCAB and toks[i + 1] in _VOCAB):
                out.append(m)
                i += 2
                continue
        seg = _split_vocab(t) if t[0].isalpha() and t not in _VOCAB and len(t) >= 6 else None
        out.extend(seg or [t])
        i += 1
    return " ".join(out)


def unsupported_numbers(answer: str, allowed_text: str, repair: bool = False) -> set[int]:
    return numbers_in(answer) - numbers_in(repair_spacing(allowed_text) if repair else allowed_text)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def _llm(schema):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        temperature=0, max_retries=3,
    ).with_structured_output(schema)


def generate(question: str, docs: list[Document], feedback: str = "") -> Draft:
    NL = chr(10)
    ctx = (NL + NL).join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    fb = (NL + NL + "CORRECTION FROM A PREVIOUS ATTEMPT: " + feedback) if feedback else ""
    return _llm(Draft).invoke(
        SYSTEM + NL + NL + "PASSAGES:" + NL + ctx + NL + NL + f"QUESTION: {question}" + fb)


class Judgement(BaseModel):
    supported: bool = Field(description="true only if EVERY factual statement in the ANSWER "
                                        "is directly stated in the EVIDENCE")
    reason: str = Field(description="one short sentence")


def judge(question: str, answer_text: str, evidence: list[str]) -> Judgement:
    NL = chr(10)
    prompt = (
        "You are a strict fact-checker. Decide whether the ANSWER is fully supported by the "
        "EVIDENCE and actually answers the QUESTION. Numbers, durations, amounts, who/what, and "
        "yes/no direction must all match the evidence. Ignore wording differences." + NL + NL
        + f"QUESTION: {question}" + NL + NL + f"ANSWER: {answer_text}" + NL + NL
        + "EVIDENCE:" + NL + (NL + NL).join(evidence))
    return _llm(Judgement).invoke(prompt)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(question: str, docs: list[Document], draft: Draft, *,
           check_numbers: bool = True, use_judge: bool = False, ref_aware: bool = True) -> Result:
    if not draft.answerable or not draft.citations:
        return Result(question, False, NOT_FOUND, sources=docs, refused_by="llm")

    # L1: quotes must exist in the cited passage ("..." joins fragments; each must appear)
    good, warnings, meta = [], [], {}
    for c in draft.citations:
        if not 1 <= c.passage <= len(docs):
            warnings.append(f"Dropped citation: passage {c.passage} does not exist.")
            continue
        d = docs[c.passage - 1]
        frags = [_norm(f) for f in re.split(r"\.{2,}|…", c.quote)]
        body = _norm(d.page_content)
        if not any(frags) or not all(f in body for f in frags if f):
            warnings.append(f"Dropped citation [{c.passage}]: quote not found in passage.")
            continue
        good.append(c)
        meta[c.passage] = d
    if not good:
        return Result(question, False, NOT_FOUND, warnings=warnings, sources=docs, refused_by="quote")

    # L2: every number in the answer must appear in a cited quote, a cited passage header,
    #     or the question itself (e.g. "section 66A").
    if check_numbers:
        allowed = " ".join(c.quote for c in good) + " " + question + " " + " ".join(
            meta[c.passage].page_content.split("\n", 1)[0] for c in good)
        text, refs = split_refs(draft.answer) if ref_aware else (draft.answer, set())
        if ref_aware:   # drop the model's own passage markers like "[2]"; they are not quantities
            text = re.sub(r"\[(\d+)\]", lambda m: " " if 1 <= int(m.group(1)) <= len(docs) else m.group(0), text)
        bad = unsupported_numbers(text, allowed, repair=ref_aware)
        # Provision references (section 15, rule 4) must exist somewhere in a CITED passage or the
        # question. Quantities above still have to be inside a quote, so a wrong amount is still caught.
        if refs:
            seen = {int(x) for x in re.findall(r"\d+", " ".join(
                [meta[c.passage].page_content for c in good] + [question]))}
            bad |= refs - seen
        if bad:
            warnings.append(f"Answer contains numbers not found in the cited quotes: {sorted(bad)}")
            return Result(question, False, NOT_FOUND, warnings=warnings, sources=docs,
                          refused_by="numbers")

    # L3: LLM judge reads answer + the cited passages (not just short quotes)
    if use_judge:
        j = judge(question, draft.answer, [meta[p].page_content for p in sorted(meta)])
        if not j.supported:
            warnings.append(f"Judge: answer not supported by evidence ({j.reason})")
            return Result(question, False, NOT_FOUND, warnings=warnings, sources=docs,
                          refused_by="judge")

    # legal-status warnings come from metadata, not from the LLM
    for c in good:
        m = meta[c.passage].metadata
        if m["status"] in ("struck_down", "superseded", "draft"):
            warnings.append(f"{m['short']} s.{m['section']} is {m['status']}: "
                            f"{m['status_note'] or 'check current status'}")
    return Result(question, True, draft.answer, good, list(dict.fromkeys(warnings)), docs)


def answer(question: str, docs: list[Document], **kw) -> Result:
    return verify(question, docs, generate(question, docs), **kw)
