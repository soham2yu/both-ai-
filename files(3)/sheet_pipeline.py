"""
Staged pipeline. Every stage reads and writes JSON in a per-exam working
directory, so any stage can be re-run — or the JSON hand-corrected — without
redoing the others:

    extract-key      key.(docx|pdf|md) [+ paper.(docx|pdf|md)]
                       -> question_paper.json, answer_key.json
    decompose        answer_key.json -> rubric.json
    extract-student  sheet.pdf       -> student_text.json
    grade            rubric.json + student_text.json -> report.json

    run              all four, in order

Conventions
  * question numbers are strings: "1", "2", ...
  * the student sheet has ONE QUESTION PER PAGE: page N answers question N
  * marks come from the answer key

Usage
    python3 sheet_pipeline.py run --key key.pdf --paper paper.md --sheet sheet.pdf --out work/exam1
"""

import argparse
import json
import os
import sys

import config
import ocr
import segmenter
from answer_key_extractor import build_answer_key, build_question_paper, cross_check
from rubric_decompose import decompose_rubric
from matcher import match_rubric_to_answer
from extra_content import find_extra_content
from report import build_report, build_sheet_report, print_sheet_summary


def _path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name)


def _write_json(out_dir: str, name: str, data) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = _path(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _read_json(out_dir: str, name: str):
    path = _path(out_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run the earlier stage that produces it first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _num_sorted(d: dict) -> list:
    return sorted(d, key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)))


# ------------------------------------------------------------------ stages ---

def stage_extract_key(key_path: str, out_dir: str, paper_path: str = None) -> dict:
    """Answer key (+ optional question paper) -> answer_key.json / question_paper.json."""
    result = build_answer_key(key_path)
    notes = list(result["notes"])
    answer_key = {n: result["answer_key"][n] for n in _num_sorted(result["answer_key"])}

    if paper_path:
        paper = build_question_paper(paper_path)
        paper = {n: paper[n] for n in _num_sorted(paper)}
        _write_json(out_dir, config.ARTIFACT_QUESTION_PAPER, paper)
        notes += cross_check(answer_key, paper)

    _write_json(out_dir, config.ARTIFACT_ANSWER_KEY, answer_key)
    return {"questions": len(answer_key), "parse_method": result["parse_method"], "notes": notes}


def stage_decompose(out_dir: str) -> dict:
    """answer_key.json -> rubric.json: {"1": {"total_marks", "points": [...]}}"""
    answer_key = _read_json(out_dir, config.ARTIFACT_ANSWER_KEY)

    missing = [n for n, v in answer_key.items() if not isinstance(v.get("marks"), (int, float))]
    if missing:
        raise ValueError(
            f"Marks missing for question(s) {', '.join(missing)} in {config.ARTIFACT_ANSWER_KEY}. "
            f"Marks are read from the answer key — add a numeric \"marks\" to each and re-run."
        )
    empty = [n for n, v in answer_key.items() if not v.get("answer", "").strip()]
    if empty:
        raise ValueError(f"Empty answer text for question(s) {', '.join(empty)} in {config.ARTIFACT_ANSWER_KEY}")

    rubric = {}
    for n in _num_sorted(answer_key):
        entry = answer_key[n]
        points = decompose_rubric(entry["answer"], entry["marks"])
        rubric[n] = {"total_marks": entry["marks"], "points": points}
    _write_json(out_dir, config.ARTIFACT_RUBRIC, rubric)
    return {"questions": len(rubric), "points": {n: len(v["points"]) for n, v in rubric.items()}}


def stage_extract_student(sheet_path: str, out_dir: str) -> dict:
    """Student PDF or scanned image -> student_text.json. Page N is question N."""
    lower = sheet_path.lower()
    if lower.endswith(".pdf"):
        pages = ocr.extract_pdf_pages(sheet_path)
    elif lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
        pages = ocr.extract_image_pages(sheet_path)
    else:
        raise ValueError("The student answer sheet must be a PDF or image file")

    student = {
        str(p["page"]): {
            "page": p["page"],
            "text": p["text"],
            "confidence": p["confidence"],
            "backend": p["backend"],
        }
        for p in pages
    }
    _write_json(out_dir, config.ARTIFACT_STUDENT_TEXT, student)

    notes = []
    rubric_path = _path(out_dir, config.ARTIFACT_RUBRIC)
    if os.path.exists(rubric_path):
        expected = set(_read_json(out_dir, config.ARTIFACT_RUBRIC))
        got = set(student)
        if expected - got:
            notes.append(f"Sheet has {len(pages)} page(s) but the rubric has question(s) "
                         f"{', '.join(_num_sorted(expected - got))} with no page — they will be graded as unanswered")
        if got - expected:
            notes.append(f"Page(s) {', '.join(_num_sorted(got - expected))} have no matching question and are ignored")
    return {"pages": len(pages), "backends": {n: v["backend"] for n, v in student.items()}, "notes": notes}


def evaluate_handwritten_sheet(key_path: str, sheet_path: str, out_dir: str, paper_path: str = None, sheet_name: str = None) -> dict:
    """High-level workflow for a handwritten or scanned answer sheet: key -> rubric -> OCR -> grade."""
    stage_extract_key(key_path, out_dir, paper_path)
    stage_decompose(out_dir)
    stage_extract_student(sheet_path, out_dir)
    return stage_grade(out_dir, sheet_name=sheet_name or os.path.basename(sheet_path))


def stage_grade(out_dir: str, sheet_name: str = None) -> dict:
    """rubric.json + student_text.json -> report.json (whole-sheet report)."""
    rubric = _read_json(out_dir, config.ARTIFACT_RUBRIC)
    student = _read_json(out_dir, config.ARTIFACT_STUDENT_TEXT)
    paper_path = _path(out_dir, config.ARTIFACT_QUESTION_PAPER)
    paper = _read_json(out_dir, config.ARTIFACT_QUESTION_PAPER) if os.path.exists(paper_path) else {}

    warnings, question_reports = [], []
    for n in _num_sorted(rubric):
        points = rubric[n]["points"]
        page = student.get(n)

        if page is None:
            text, confidence, backend, page_no = "", 0.0, "no_page", None
            warnings.append(f"Q{n}: no page found in the student sheet — graded as unanswered")
        else:
            text, confidence, backend, page_no = page["text"], page["confidence"], page["backend"], page["page"]

        # sentence segmentation happens inside the matcher / extra-content detector
        # (both go through segmenter.segment_sentences); recorded here for the reviewer
        sentences = segmenter.segment_sentences(text)

        matches = match_rubric_to_answer(points, text)
        extras = find_extra_content(points, text)

        report = build_report(n, matches, ocr_confidence=confidence)
        if not text.strip():
            report["needs_human_review"] = True
            report["review_reasons"].append("no text extracted for this question")
        report["extra_content_flags"] = extras
        report["question_text"] = paper.get(n, {}).get("question")
        report["student_page"] = page_no
        report["ocr_backend"] = backend
        report["student_sentences"] = sentences
        question_reports.append(report)

    answered_but_unrubriced = [n for n in student if n not in rubric]
    for n in _num_sorted(answered_but_unrubriced):
        warnings.append(f"Page {n} has no matching question in the rubric — ignored")

    sheet_report = build_sheet_report(question_reports, {"name": sheet_name, "warnings": warnings})
    _write_json(out_dir, config.ARTIFACT_REPORT, sheet_report)
    return sheet_report


# --------------------------------------------------------------------- CLI ---

def _print_notes(label: str, notes: list):
    for n in notes:
        print(f"  ! {label}: {n}")


def run_all(key_path, sheet_path, out_dir, paper_path=None) -> dict:
    r = stage_extract_key(key_path, out_dir, paper_path)
    print(f"[1/4] answer key: {r['questions']} question(s) via {r['parse_method']}")
    _print_notes("key", r["notes"])

    r = stage_decompose(out_dir)
    print(f"[2/4] rubric: points per question {r['points']}")

    r = stage_extract_student(sheet_path, out_dir)
    print(f"[3/4] student sheet: {r['pages']} page(s), extraction {r['backends']}")
    _print_notes("sheet", r["notes"])

    report = stage_grade(out_dir, sheet_name=os.path.basename(sheet_path))
    print("[4/4] graded\n")
    print_sheet_summary(report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Staged exam-grading pipeline (file-based artifacts)")
    sub = p.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("extract-key", help="answer key (+ question paper) -> JSON")
    k.add_argument("--key", required=True); k.add_argument("--paper"); k.add_argument("--out", required=True)

    d = sub.add_parser("decompose", help="answer_key.json -> rubric.json")
    d.add_argument("--out", required=True)

    s = sub.add_parser("extract-student", help="student PDF -> student_text.json")
    s.add_argument("--sheet", required=True); s.add_argument("--out", required=True)

    g = sub.add_parser("grade", help="rubric + student text -> report.json")
    g.add_argument("--out", required=True); g.add_argument("--name")

    r = sub.add_parser("run", help="all stages")
    r.add_argument("--key", required=True); r.add_argument("--paper")
    r.add_argument("--sheet", required=True); r.add_argument("--out", required=True)

    a = p.parse_args(argv)
    try:
        if a.cmd == "extract-key":
            res = stage_extract_key(a.key, a.out, a.paper); print(json.dumps(res, indent=2))
        elif a.cmd == "decompose":
            res = stage_decompose(a.out); print(json.dumps(res, indent=2))
        elif a.cmd == "extract-student":
            res = stage_extract_student(a.sheet, a.out); print(json.dumps(res, indent=2))
        elif a.cmd == "grade":
            print_sheet_summary(stage_grade(a.out, a.name))
        elif a.cmd == "run":
            run_all(a.key, a.sheet, a.out, a.paper)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
