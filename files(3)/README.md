# Exam grading pipeline (Ollama-only, file-based stages)

Question paper + answer key + a student's scanned answer sheet in →
one whole-sheet report out. Fully local: Tesseract, spaCy and Ollama.

## Stages

Each stage reads/writes JSON in a working directory, so any stage can be
re-run, or its JSON hand-corrected, without redoing the others.

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 1 Extract key | `extract-key --key K [--paper P]` | answer key (.docx/.pdf/.md), question paper | `answer_key.json`, `question_paper.json` |
| 2 Decompose | `decompose` | `answer_key.json` | `rubric.json` |
| 3 Extract student text | `extract-student --sheet S.pdf` | student PDF | `student_text.json` |
| 4 Grade | `grade` | rubric + student text | `report.json` |

```bash
python3 sheet_pipeline.py run --key key.pdf --paper paper.md --sheet sheet.pdf --out work/exam1
# or stage by stage:
python3 sheet_pipeline.py extract-key --key key.pdf --paper paper.md --out work/exam1
python3 sheet_pipeline.py decompose --out work/exam1
python3 sheet_pipeline.py extract-student --sheet sheet.pdf --out work/exam1
python3 sheet_pipeline.py grade --out work/exam1
```

## Conventions (assumptions the code relies on)

- Question numbers are strings: `"1"`, `"2"`, …
- **Student sheet: one question per page — page N answers question N.**
  Cover pages, optional questions and answers spanning pages are not handled.
- **Marks come from the answer key.** A missing mark is left `null` and
  `decompose` refuses to run until you fill it in — it never guesses.
- Answer key input: `.docx`, `.pdf` (typed or scanned), `.md`/`.txt`.
  Legacy `.doc` must be converted to `.docx`.

## Artifact shapes

```jsonc
// answer_key.json
{"1": {"answer": "Newton's third law states …", "marks": 6.0}}
// rubric.json
{"1": {"total_marks": 6.0, "points": [{"id": "r1", "point": "…", "marks": 3.0}]}}
// student_text.json
{"1": {"page": 1, "text": "…", "confidence": 1.0, "backend": "text_layer"}}
// report.json → {"sheet", "summary": {total_awarded, total_possible, percentage,
//                 questions_needing_review, warnings}, "questions": [per-question report]}
```

## How each stage decides (cheapest first)

**Student text, per page:** PDF text layer (no OCR) → Tesseract, kept if mean
word confidence ≥ `OCR_TESSERACT_MIN_CONFIDENCE` (0.65) → Ollama vision
otherwise → if Ollama can't help, keep Tesseract's text with confidence capped
so the page is flagged for review. `backend` records which path ran.

**Sentences:** `segmenter.py` repairs OCR line-wraps/hyphenation and keeps
list items separate before spaCy splits sentences.

**Rubric:** spaCy clause split (verb+verb "and", sentences, `;`, list items;
list lead-ins like "Steps:" are dropped) → embedding split check for still-long
chunks. Marks split evenly.

**Judging, per rubric point:** TF-IDF → Ollama embeddings → Ollama LLM, each
tier only if the previous one was ambiguous. Evidence quotes are verified
against the student text. Sentences matching no rubric point are surfaced as
extra content. `resolved_by` in each result says which tier decided.

**Review flag:** any low-confidence point, many partial matches, low OCR
confidence, or no text extracted → `needs_human_review`; the sheet summary
lists which questions.

## Setup

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# Tesseract ENGINE (pytesseract is only a wrapper)
sudo apt-get install tesseract-ocr     # macOS: brew install tesseract

# Ollama (https://ollama.com) — optional but needed for the vision/embedding/LLM tiers
ollama pull llama3.1:8b
ollama pull llava
ollama pull nomic-embed-text
python3 llm_client.py        # shows which of these are actually present
```

Without Ollama the pipeline still runs end to end; unresolved items come out
as lower confidence / `needs_human_review` rather than as errors.
Every threshold and model name is in `config.py` and overridable by env var.

## Tests

`python3 test_sheet_pipeline.py` — builds its own fixtures (markdown key,
question paper, a 3-page PDF with a text-layer page, a clean scan and a
degraded scan) and checks every stage, the OCR escalation logic (Ollama
mocked), and the failure modes.

## Not verified / known limits

- **Nothing here has run against a real Ollama.** Vision OCR, embeddings and
  LLM tie-breaking are tested only via mocks and in their degraded mode.
  Small vision models (llava) are weak on handwriting — benchmark on real
  scans; the model is a one-line config change.
- `OCR_TESSERACT_MIN_CONFIDENCE` is untuned. Tesseract on real handwriting
  often returns wrong text at a middling confidence, which would skip the
  Ollama escalation. Check against real scans.
- Partial credit is a flat 50% and marks are split evenly across points.
- `contradicted` can only come from the LLM tier.
- A numbered list inside an answer key with *un*-prefixed questions
  (`1.`, `2.`) can be mistaken for question markers; `Q1`-style markers are
  robust to this.
- Diagram/maths answers are not handled.

## Older demo layer

`api.py` + `frontend/` (FastAPI + plain JS) and `main_pipeline.py` still
work on the earlier metadata-driven flow (per-question page mapping, typed
reference answers). They have **not** been rebuilt around the staged
pipeline above. Run: `uvicorn api:app --port 8000`.
