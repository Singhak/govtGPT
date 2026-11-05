"""Step 2b: PDF -> clean text -> Section/Rule units -> chunks with metadata.

Pipeline:  load_pages -> clean -> split_into_units -> chunk_unit -> Documents
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# 1. Config: har document ka format alag hai, isliye per-document config.
# ---------------------------------------------------------------------------


@dataclass
class DocConfig:
    file: str
    act: str                 # full name (citation me dikhega)
    short: str               # chhota naam, e.g. "IT Act 2000"
    year: int
    doc_type: str            # "act" | "rules"
    unit: str                # "Section" | "Rule"
    status: str              # "enacted" | "draft" ...
    require_dash: bool = False   # heading ke baad "—"/"–" hona zaroori? (ToC filter)
    note: str = ""
    # {"66A": "struck_down", ...} - sirf un sections ke liye jinki legal status alag hai
    overrides: dict = field(default_factory=dict)


CONFIGS = [
    DocConfig(
        file="it_act_2000.pdf",
        act="The Information Technology Act, 2000",
        short="IT Act 2000", year=2000, doc_type="act", unit="Section",
        status="enacted", require_dash=True,
        note="India Code copy, includes 2008 amendments",
        overrides={
            "66A": {
                "status": "struck_down",
                "note": "Struck down by Supreme Court in Shreya Singhal v. UoI (2015); "
                        "still printed in the text but not enforceable.",
            },
            "43A": {
                "status": "superseded",
                "note": "DPDP Act 2023 s.44(2) provides for omission of s.43A; "
                        "check the commencement notification for the date.",
            },
        },
    ),
    DocConfig(
        file="it_rules_2021.pdf",
        act="The Information Technology (Intermediary Guidelines and Digital Media "
            "Ethics Code) Rules, 2021",
        short="IT Rules 2021", year=2021, doc_type="rules", unit="Rule",
        status="enacted",
        note="Original Feb-2021 notification; later amendments (e.g. 2022) not included",
    ),
    DocConfig(
        file="dpdp_act_2023.pdf",
        act="The Digital Personal Data Protection Act, 2023",
        short="DPDP Act 2023", year=2023, doc_type="act", unit="Section",
        status="enacted",
        note="Section titles are marginal notes in the PDF and are not extracted",
    ),
    DocConfig(
        file="dpdp_rules_2025.pdf",
        act="The Digital Personal Data Protection Rules, 2025",
        short="DPDP Rules 2025", year=2025, doc_type="rules", unit="Rule",
        status="enacted",
        note="English pages only (Hindi half has broken font encoding); "
             "commencement is phased",
    ),
]

# ---------------------------------------------------------------------------
# 2. Loading + cleaning
# ---------------------------------------------------------------------------

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Running header/footer noise jo har page par repeat hota hai
_NOISE_LINE = re.compile(
    r"GAZETTE OF INDIA|^\s*SEC\.\s*\d+\]|^\s*\d{1,3}\s*$|REGD\. No|xxxGID|^\s*GI/\d{4}",
    re.I,
)
# IT Act ke page-bottom footnotes: "2. Ins. by Act 7 of 2017 ..." / "5*   *   *"
_FOOTNOTE = re.compile(
    r"^\s*\d{1,2}\.\s+(Ins|Subs|Omitted|Clause|Clauses|The (word|words|first|second|"
    r"proviso|figure)|Added|Rep|Sub-section|Explanation)\b|^\s*\d{1,2}\*",
)


def load_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_no, cleaned_text)], Hindi pages/lines aur noise hata kar."""
    out = []
    for i, page in enumerate(PdfReader(pdf_path).pages, start=1):
        raw = page.extract_text() or ""
        hindi, latin = len(_DEVANAGARI.findall(raw)), len(_LATIN.findall(raw))
        if hindi > latin:          # poora Hindi page (font garbled) -> skip
            continue
        lines = []
        for line in raw.splitlines():
            if _DEVANAGARI.search(line):      # Hindi header/lines
                continue
            if _NOISE_LINE.search(line) or _FOOTNOTE.search(line):
                continue
            lines.append(line.rstrip())
        out.append((i, "\n".join(lines)))
    return out


def _normalise(text: str) -> str:
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    return text


# ---------------------------------------------------------------------------
# 3. Split into legal units (Section / Rule / Schedule)
# ---------------------------------------------------------------------------

# Amended sections PDF me "3[43A. Compensation..." ki tarah shuru hote hain
# (footnote number + "[" pehle lagta hai), isliye optional prefix allow hai.
_HEADING = re.compile(r"^\s*(?:\d{1,2}\[)?(\d{1,3})([A-Z]?)\.\s+(?=\S)")
_SCHEDULE = re.compile(
    r"^\s*(THE\s+)?((FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)\s+)?SCHEDULE\s*$"
)
_TITLE_END = re.compile(r"\s*[.:]?\s*[—–-]|:")


@dataclass
class Unit:
    number: str          # "43A" / "3" / "SCHEDULE"
    title: str
    text: str
    page: int
    is_schedule: bool = False


def _is_next(prev: tuple[int, str] | None, n: int, letter: str) -> bool:
    """Monotonic rule: section numbers sirf aage badhte hain.

    Isse footnotes ("3. Ins. by..."), schedule-table items ("1.", "2.") aur
    ToC jaise false headings reject ho jaate hain. Omitted sections ki wajah se
    kuch numbers skip ho sakte hain, isliye +5 tak ka gap allowed hai.
    """
    if prev is None:
        return n == 1 and letter == ""
    pn, pl = prev
    if n == pn:
        return letter > pl               # 66 -> 66A, 66A -> 66B
    return pn < n <= pn + 5


def split_into_units(pages: list[tuple[int, str]], cfg: DocConfig) -> list[Unit]:
    units: list[Unit] = []
    cur: dict | None = None
    prev: tuple[int, str] | None = None
    in_schedule = False

    def flush():
        nonlocal cur
        if cur and cur["lines"]:
            units.append(Unit(cur["number"], cur["title"],
                              "\n".join(cur["lines"]).strip(), cur["page"],
                              cur["sched"]))
        cur = None

    for page_no, text in pages:
        lines = [ln for ln in _normalise(text).splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            # Schedule header? (sirf tab jab pehla section mil chuka ho -> ToC ignore)
            if prev is not None and _SCHEDULE.match(line.strip()):
                flush()
                in_schedule = True
                cur = {"number": "SCHEDULE", "title": line.strip(), "lines": [],
                       "page": page_no, "sched": True}
                continue

            m = None if in_schedule else _HEADING.match(line)
            if m:
                n, letter = int(m.group(1)), m.group(2)
                rest = line[m.end():]
                title_m = _TITLE_END.search(rest)
                has_dash = bool(re.search(r"[—–-]", rest[:200]))
                if not has_dash:
                    # Long titles wrap: "67B. Punishment for ... in" / "form.–(1) Whoever"
                    ahead = " ".join(lines[idx + 1: idx + 3])[:250]
                    has_dash = bool(re.search(r"\.\s*[—–]", ahead))
                ok = _is_next(prev, n, letter) and (has_dash or not cfg.require_dash)
                if ok:
                    flush()
                    prev = (n, letter)
                    title = rest[: title_m.start()].strip(" .:—–-") if title_m else ""
                    if len(title) > 150:      # title nahi, poora vaakya hai
                        title = ""
                    cur = {"number": f"{n}{letter}", "title": title,
                           "lines": [line.strip()], "page": page_no, "sched": False}
                    continue
            if cur is None:                    # preamble / ToC (section 1 se pehle)
                continue
            cur["lines"].append(line.strip())
    flush()
    return units


# ---------------------------------------------------------------------------
# 4. Chunking + metadata
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1800      # characters (~400-450 tokens)
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # pehle sub-section/clause boundary par todo, phir line, phir shabd
    separators=["\n(", "\n", ". ", " "],
    keep_separator="start",
)


def chunk_unit(unit: Unit, cfg: DocConfig) -> list[Document]:
    label = "Schedule" if unit.is_schedule else f"{cfg.unit} {unit.number}"
    if unit.is_schedule and unit.title:
        label = unit.title.title()
    ov = {} if unit.is_schedule else cfg.overrides.get(unit.number, {})
    # Contextual header: har chunk ke text me bhi source likha jaata hai,
    # taaki embedding aur LLM dono ko pata rahe ye chunk kahaan se hai.
    header = f"[{cfg.short} | {label}" + (f": {unit.title}" if unit.title else "") + "]"
    if ov.get("status"):
        header += f" (status: {ov['status']})"

    pieces = (
        [unit.text] if len(unit.text) <= CHUNK_SIZE else _splitter.split_text(unit.text)
    )
    docs = []
    for idx, piece in enumerate(pieces):
        docs.append(Document(
            page_content=f"{header}\n{piece}",
            metadata={
                "act": cfg.act,
                "short": cfg.short,
                "year": cfg.year,
                "doc_type": cfg.doc_type,
                "unit": "Schedule" if unit.is_schedule else cfg.unit,
                "section": unit.number,
                "title": unit.title,
                "page": unit.page,
                "chunk_index": idx,
                "chunk_total": len(pieces),
                "status": ov.get("status", cfg.status),
                "status_note": ov.get("note", ""),
                "source_file": cfg.file,
            },
        ))
    return docs


def ingest(data_dir: str | Path = "data") -> list[Document]:
    data_dir = Path(data_dir)
    all_docs: list[Document] = []
    for cfg in CONFIGS:
        pages = load_pages(data_dir / cfg.file)
        units = split_into_units(pages, cfg)
        for u in units:
            all_docs.extend(chunk_unit(u, cfg))
    return all_docs
