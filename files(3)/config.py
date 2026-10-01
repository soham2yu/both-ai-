"""
Central configuration — every tunable value in the pipeline lives here,
sourced from environment variables with sensible defaults. No other module
should hardcode a model name, threshold, URL, or file path; import from
here instead.

Why this matters beyond tidiness: model names go stale (Gemini model IDs
get deprecated), thresholds need tuning per subject/institution without a
code change, and a hackathon demo often needs a last-minute override
("the wifi is bad, force Ollama-only") that should be one env var, not a
grep-and-edit across six files.

Override any of these at runtime, e.g.:
    export MATCHER_HIGH_THRESHOLD=0.5
    export GEMINI_TEXT_MODEL=gemini-2.0-flash
    python3 main_pipeline.py
"""

import os


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# LLM backend (llm_client.py) -- Ollama only, fully local
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "llama3.1:8b")
# Qwen2.5-VL handles document layout and handwriting better than the older
# LLaVA fallback on this exam dataset.
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

LLM_GENERATE_TIMEOUT_SECONDS = _int_env("LLM_GENERATE_TIMEOUT_SECONDS", 60)
OLLAMA_VISION_TIMEOUT_SECONDS = _int_env("OLLAMA_VISION_TIMEOUT_SECONDS", 600)  # CPU vision on a full page is slow
LLM_HEALTHCHECK_TIMEOUT_SECONDS = _float_env("LLM_HEALTHCHECK_TIMEOUT_SECONDS", 1.5)
OLLAMA_STATUS_CACHE_SECONDS = _float_env("OLLAMA_STATUS_CACHE_SECONDS", 10.0)  # avoid a health-check GET before every single call

# Force-disable Ollama regardless of availability -- handy for proving the
# pipeline degrades cleanly ("run it with the LLM tiers off").
DISABLE_OLLAMA = os.environ.get("DISABLE_OLLAMA", "false").lower() == "true"

# ---------------------------------------------------------------------------
# OCR (ocr.py)
# ---------------------------------------------------------------------------
# A PDF page whose embedded text layer has at least this many characters is
# treated as machine-readable and NOT sent to OCR at all.
PDF_TEXT_LAYER_MIN_CHARS = _int_env("PDF_TEXT_LAYER_MIN_CHARS", 30)
# Tesseract's mean word confidence (0-1). Below this, the page escalates to
# Ollama vision. Tesseract on handwriting tends to return garbage at a
# middling confidence rather than failing, so tune this on real scans.
OCR_TESSERACT_MIN_CONFIDENCE = _float_env("OCR_TESSERACT_MIN_CONFIDENCE", 0.65)
# Ollama vision models report no per-call confidence, so this is a flat
# assumed value for its output.
OCR_OLLAMA_VISION_CONFIDENCE = _float_env("OCR_OLLAMA_VISION_CONFIDENCE", 0.7)
# If a page needed vision but Ollama couldn't provide it, Tesseract's text is
# kept but its confidence is capped here so the page still gets flagged.
OCR_LOW_CONFIDENCE_CAP = _float_env("OCR_LOW_CONFIDENCE_CAP", 0.5)
VISION_MAX_SIDE_PX = _int_env("VISION_MAX_SIDE_PX", 2000)

# ---------------------------------------------------------------------------
# PDF ingestion (pdf_ingest.py)
# ---------------------------------------------------------------------------
PDF_RENDER_DPI = _int_env("PDF_RENDER_DPI", 300)  # bump to 400+ if OCR quality is poor on handwriting

# ---------------------------------------------------------------------------
# spaCy / rubric decomposition (rubric_decompose.py)
# ---------------------------------------------------------------------------
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
RUBRIC_COMPOUND_WORD_COUNT = _int_env("RUBRIC_COMPOUND_WORD_COUNT", 18)
RUBRIC_EMBED_SPLIT_ACCEPT_THRESHOLD = _float_env("RUBRIC_EMBED_SPLIT_ACCEPT_THRESHOLD", 0.55)

# ---------------------------------------------------------------------------
# Matching / diff engine (matcher.py)
# ---------------------------------------------------------------------------
MATCHER_TFIDF_HIGH_THRESHOLD = _float_env("MATCHER_TFIDF_HIGH_THRESHOLD", 0.45)
MATCHER_TFIDF_LOW_THRESHOLD = _float_env("MATCHER_TFIDF_LOW_THRESHOLD", 0.15)
MATCHER_EMBED_HIGH_THRESHOLD = _float_env("MATCHER_EMBED_HIGH_THRESHOLD", 0.62)
MATCHER_EMBED_LOW_THRESHOLD = _float_env("MATCHER_EMBED_LOW_THRESHOLD", 0.35)
MATCHER_QUOTE_VERIFY_MIN_RATIO = _float_env("MATCHER_QUOTE_VERIFY_MIN_RATIO", 0.75)
MATCHER_PARTIAL_CREDIT_FRACTION = _float_env("MATCHER_PARTIAL_CREDIT_FRACTION", 0.5)
ONE_LINER_MATCH_THRESHOLD = _float_env("ONE_LINER_MATCH_THRESHOLD", 0.55)
NUMERIC_TOLERANCE_FRACTION = _float_env("NUMERIC_TOLERANCE_FRACTION", 0.02)

# ---------------------------------------------------------------------------
# Extra-content detection (extra_content.py)
# ---------------------------------------------------------------------------
EXTRA_CONTENT_UNCOVERED_THRESHOLD = _float_env("EXTRA_CONTENT_UNCOVERED_THRESHOLD", 0.12)
EXTRA_CONTENT_MIN_WORDS = _int_env("EXTRA_CONTENT_MIN_WORDS", 5)

# ---------------------------------------------------------------------------
# Report / human-review flagging (report.py)
# ---------------------------------------------------------------------------
REVIEW_CONFIDENCE_THRESHOLD = _float_env("REVIEW_CONFIDENCE_THRESHOLD", 0.5)
REVIEW_PARTIAL_FRACTION = _float_env("REVIEW_PARTIAL_FRACTION", 0.3)
REVIEW_MIN_OCR_CONFIDENCE = _float_env("REVIEW_MIN_OCR_CONFIDENCE", 0.6)

# ---------------------------------------------------------------------------
# Data file locations (main_pipeline.py) -- faculty answers and page-mapping
# metadata are real institutional data, not something to hardcode into
# Python. These point at example JSON files by default; override the paths
# (via env var or CLI arg -- see main_pipeline.py --help) to point at your
# actual data.
# ---------------------------------------------------------------------------
FACULTY_ANSWERS_PATH = os.environ.get("FACULTY_ANSWERS_PATH", "faculty_answers.example.json")
PAGE_METADATA_PATH = os.environ.get("PAGE_METADATA_PATH", "page_metadata.example.json")

# ---------------------------------------------------------------------------
# Staged pipeline artifacts (sheet_pipeline.py) -- one JSON file per stage,
# all written into a per-exam working directory.
# ---------------------------------------------------------------------------
ARTIFACT_QUESTION_PAPER = os.environ.get("ARTIFACT_QUESTION_PAPER", "question_paper.json")
ARTIFACT_ANSWER_KEY = os.environ.get("ARTIFACT_ANSWER_KEY", "answer_key.json")
ARTIFACT_RUBRIC = os.environ.get("ARTIFACT_RUBRIC", "rubric.json")
ARTIFACT_STUDENT_TEXT = os.environ.get("ARTIFACT_STUDENT_TEXT", "student_text.json")
ARTIFACT_REPORT = os.environ.get("ARTIFACT_REPORT", "report.json")


if __name__ == "__main__":
    import json
    print(json.dumps({k: v for k, v in globals().items() if k.isupper()}, indent=2, default=str))
