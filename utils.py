import os
import re
import math
import functools
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

load_dotenv()

# Must match the cap enforced inside agents/nodes.py (_story), so the sampled text
# is never silently truncated by the pipeline.
MAX_STORY_CHARS = int(os.getenv("MAX_STORY_CHARS", "60000"))

KEY_EVENT_SEPARATOR = "\n\n...[KEY EVENT EXTRACTED]...\n\n"


# =====================================================================
# 1. File decoding
# =====================================================================
def decode_text(raw: bytes) -> str:
    """Decodes an uploaded .txt robustly (UTF-8 with/without BOM, then cp1252) and normalizes newlines."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# =====================================================================
# 2. NLTK resources (lazy, with offline fallbacks)
# =====================================================================
_WORD_RE = re.compile(r"[a-z]+")  # alphabetic tokens only: no quote marks, "n't", numbers, punctuation
_MIN_CONTENT_WORDS = 5            # ignore fragments like "Yes." when ranking sentences

# Dialogue tags / filler that would otherwise dominate frequency scoring in fiction.
_EXTRA_STOPWORDS = frozenset({"said", "asked", "replied", "says", "say", "would", "could", "upon"})
_FALLBACK_STOPWORDS = frozenset(
    "a an the and or but if then so of to in on at by for with from as is are was were be been being "
    "i me my we our you your he him his she her it its they them their this that these those what which "
    "who whom do does did have has had not no nor too very can will just than s t d ll m re ve".split()
)


def _try_nltk(probe, downloads: Iterable[str]):
    """Runs probe(); on a missing resource, downloads it once and retries. Returns None on failure."""
    try:
        return probe()
    except LookupError:
        pass
    except Exception:
        return None
    for package in downloads:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    try:
        return probe()
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _nltk_sentences_available() -> bool:
    # Newer NLTK needs 'punkt_tab'; older versions need 'punkt'. Fetch whichever is missing.
    return _try_nltk(lambda: sent_tokenize("One. Two."), ("punkt_tab", "punkt")) is not None


@functools.lru_cache(maxsize=None)
def _stopword_set() -> frozenset:
    words = _try_nltk(lambda: set(stopwords.words("english")), ("stopwords",))
    return frozenset((words or _FALLBACK_STOPWORDS) | _EXTRA_STOPWORDS)


_FALLBACK_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"'”’]?\s+(?=[\"'“‘]?[A-Z])")


def _split_sentences(text: str) -> List[str]:
    if _nltk_sentences_available():
        return sent_tokenize(text)
    return [s for s in _FALLBACK_SENTENCE_SPLIT.split(text) if s.strip()]


# =====================================================================
# 3. Compression & sampling of long stories
# =====================================================================
def compress_middle_text_nltk(text: str, num_sentences: int = 50) -> str:
    """Extractively keeps the most informative sentences of a large text block, in original order."""
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return text  # already short

    stop_words = _stopword_set()
    tokenized = [
        [w for w in _WORD_RE.findall(sentence.lower()) if w not in stop_words]
        for sentence in sentences
    ]

    frequencies = Counter(w for tokens in tokenized for w in tokens)
    max_freq = max(frequencies.values(), default=1)

    # Score = summed normalized word frequency, damped by sqrt(length) so that very long
    # sentences don't win automatically.
    scored: List[Tuple[float, int]] = []
    for i, tokens in enumerate(tokenized):
        if len(tokens) < _MIN_CONTENT_WORDS:
            continue
        score = sum(frequencies[w] / max_freq for w in tokens) / math.sqrt(len(tokens))
        scored.append((-score, i))
    scored.sort()  # best score first; ties broken by position (deterministic)

    top_indices = sorted(i for _, i in scored[:num_sentences])
    if not top_indices:  # degenerate text: fall back to evenly spaced sentences
        step = max(1, len(sentences) // num_sentences)
        top_indices = list(range(0, len(sentences), step))[:num_sentences]

    return KEY_EVENT_SEPARATOR.join(sentences[i] for i in top_indices)


def _snap_head(text: str, limit: int) -> str:
    """First `limit` chars, trimmed back to a paragraph/sentence boundary when one is nearby."""
    head = text[:limit]
    window_start = max(0, len(head) - 600)
    cut = max(head.rfind("\n", window_start), head.rfind(". ", window_start))
    return head[:cut + 1] if cut > 0 else head


def _snap_tail(text: str, limit: int) -> str:
    """Last `limit` chars, trimmed forward to a paragraph/sentence boundary when one is nearby."""
    tail = text[-limit:]
    candidates = [i for i in (tail.find("\n", 0, 600), tail.find(". ", 0, 600)) if i != -1]
    return tail[min(candidates) + 1:].lstrip() if candidates else tail


def sample_story_text(
    story_text: str,
    direct_limit: int = 40000,
    head_chars: int = 20000,
    tail_chars: int = 12000,
    num_sentences: int = 75,
    max_chars: int = MAX_STORY_CHARS,
) -> str:
    """
    Returns the text to feed the LangGraph pipeline. Short stories pass through unchanged;
    long ones keep the intro and conclusion verbatim and compress the middle. The result is
    kept under `max_chars` so agents/nodes.py never has to truncate it.
    """
    if len(story_text) <= direct_limit:
        return story_text

    head = _snap_head(story_text, head_chars)
    tail = _snap_tail(story_text, tail_chars)
    middle = story_text[len(head): len(story_text) - len(tail)]
    compressed = compress_middle_text_nltk(middle, num_sentences=num_sentences)

    overhead = 120  # section markers
    budget = max_chars - len(head) - len(tail) - overhead
    if budget <= 0:
        compressed = ""
    elif len(compressed) > budget:
        kept, used = [], 0
        for part in compressed.split(KEY_EVENT_SEPARATOR):
            cost = len(part) + len(KEY_EVENT_SEPARATOR)
            if used + cost > budget:
                break
            kept.append(part)
            used += cost
        compressed = KEY_EVENT_SEPARATOR.join(kept)

    return (
        f"{head}\n\n--- COMPRESSED MIDDLE TIMELINE ---\n\n{compressed}"
        f"\n\n--- CONCLUSION ---\n\n{tail}"
    )


# =====================================================================
# 4. Chapter detection for Qdrant metadata
# =====================================================================
# Only real headings count (start of a line), never in-text references like "see chapter 3".
_CHAPTER_RE = re.compile(r"^[ \t]*(?i:chapter)[ \t]+(\d+|[IVXLCDM]+)\b", flags=re.MULTILINE)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _chapter_token_to_int(token: str) -> int:
    if token.isdigit():
        return int(token)
    total, prev = 0, 0
    for ch in reversed(token.upper()):
        value = _ROMAN[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def update_chapter_number(chunk: str, current: int) -> int:
    """
    Returns the chapter number that applies to `chunk`. Reads the number from the last heading
    in the chunk, so overlapping chunks that repeat a heading don't double-count, and the first
    "Chapter 1" is labelled 1 (not 2).
    """
    matches = _CHAPTER_RE.findall(chunk)
    if not matches:
        return current
    number = _chapter_token_to_int(matches[-1])
    return number if number > 0 else current


# =====================================================================
# 5. Pipeline progress (driven by LangGraph "updates" events)
# =====================================================================
class PipelineProgress:
    """
    Tracks the 4 UI stages in pipeline order: planner -> analyzer -> cartographer -> synthesizer.
    A stage only turns "done" when its verifier node approves it (a retry keeps it "active"),
    and the next stage then becomes active.
    """
    STAGES = ("planner", "analyzer", "cartographer", "synthesizer")

    def __init__(self):
        self.status: Dict[str, str] = {stage: "pending" for stage in self.STAGES}
        self.status[self.STAGES[0]] = "active"

    def update(self, node_name: str, node_update: Optional[dict]) -> bool:
        """Feeds one (node, update) pair from the stream. Returns True if the display changed."""
        if not node_name.startswith("verify_"):
            return False
        stage = node_name[len("verify_"):]
        proceed = ((node_update or {}).get("proceed") or {}).get(stage)
        if proceed is not True or stage not in self.status:
            return False

        unverified = (node_update or {}).get("unverified_sections") or []
        llm_err = ((node_update or {}).get("llm_errors") or {}).get(stage)
        if stage in unverified or llm_err is not None:
            self.status[stage] = "failed"
        else:
            self.status[stage] = "done"

        position = self.STAGES.index(stage)
        if position + 1 < len(self.STAGES):
            self.status[self.STAGES[position + 1]] = "active"
        return True

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return tuple(self.status[stage] for stage in self.STAGES)  # type: ignore[return-value]