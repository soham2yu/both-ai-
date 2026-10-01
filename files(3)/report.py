"""
Report generator — assembles the final per-question JSON your teammates'
UI/overlay module will consume. This is the interface contract: agree this
shape with whoever builds the report/visual-overlay UI, then you can each
build independently against it.
"""

import json
import config

REVIEW_CONFIDENCE_THRESHOLD = config.REVIEW_CONFIDENCE_THRESHOLD   # below this, per-point confidence triggers human review
REVIEW_PARTIAL_FRACTION = config.REVIEW_PARTIAL_FRACTION            # if this fraction of points are "partial", flag whole answer for review


def build_report(question_id: str, rubric_results: list, ocr_confidence: float = None) -> dict:
    total_possible = sum(r["marks_allocated"] for r in rubric_results)
    total_awarded = sum(r["marks_awarded"] for r in rubric_results)

    low_confidence_points = [r for r in rubric_results if r["confidence"] < REVIEW_CONFIDENCE_THRESHOLD]
    partial_points = [r for r in rubric_results if r["status"] == "partial"]

    needs_human_review = (
        len(low_confidence_points) > 0
        or (len(rubric_results) > 0 and len(partial_points) / len(rubric_results) > REVIEW_PARTIAL_FRACTION)
        or (ocr_confidence is not None and ocr_confidence < config.REVIEW_MIN_OCR_CONFIDENCE)
    )

    review_reasons = []
    if low_confidence_points:
        review_reasons.append(f"{len(low_confidence_points)} rubric point(s) below confidence threshold")
    if len(rubric_results) > 0 and len(partial_points) / len(rubric_results) > REVIEW_PARTIAL_FRACTION:
        review_reasons.append("high proportion of partial matches")
    if ocr_confidence is not None and ocr_confidence < config.REVIEW_MIN_OCR_CONFIDENCE:
        review_reasons.append("low OCR confidence on source scan")

    return {
        "question_id": question_id,
        "rubric_results": [
            {
                "point": r["point"],
                "marks_allocated": r["marks_allocated"],
                "status": r["status"],
                "evidence_quote": r["evidence_quote"],
                "confidence": r["confidence"],
                "marks_awarded": r["marks_awarded"],
                "quote_verified": r["quote_verified"],
                "resolved_by": r.get("resolved_by"),  # which tier made the call: tfidf / embeddings / llm / ...
            }
            for r in rubric_results
        ],
        "total_awarded": round(total_awarded, 2),
        "total_possible": round(total_possible, 2),
        "needs_human_review": needs_human_review,
        "review_reasons": review_reasons,
        "ocr_confidence": ocr_confidence,
    }


def print_human_summary(report: dict):
    """Readable console summary -- useful for judging/demo narration."""
    print(f"\nQuestion: {report['question_id']}")
    print(f"Score: {report['total_awarded']} / {report['total_possible']}")
    print(f"Needs human review: {report['needs_human_review']}", end="")
    if report["review_reasons"]:
        print(f"  ({'; '.join(report['review_reasons'])})")
    else:
        print()
    print("-" * 60)
    for r in report["rubric_results"]:
        icon = {"matched": "✅", "partial": "⚠️ ", "missing": "❌", "contradicted": "🚫"}[r["status"]]
        print(f"{icon} [{r['marks_awarded']}/{r['marks_allocated']}] {r['point']}")
        if r["evidence_quote"]:
            print(f"     evidence: \"{r['evidence_quote']}\" (confidence {r['confidence']}, verified={r['quote_verified']})")
        else:
            print(f"     no supporting evidence found in answer")
    print("-" * 60)


if __name__ == "__main__":
    from rubric_decompose import decompose_rubric
    from matcher import match_rubric_to_answer

    ref_answer = "Newton's third law states that for every action there is an equal and opposite reaction; forces always occur in pairs acting on two different bodies."
    rubric = decompose_rubric(ref_answer, total_marks=6)

    student_answer = "Newton's 3rd law says every action has an equal and opposite reaction. I also like physics a lot."
    results = match_rubric_to_answer(rubric, student_answer)

    report = build_report("q3", results, ocr_confidence=0.91)
    print(json.dumps(report, indent=2))
    print_human_summary(report)


# ---------------------------------------------------------------------------
# Whole-answer-sheet report
# ---------------------------------------------------------------------------

def _qsort_key(question_id: str):
    try:
        return (0, int(question_id))
    except (TypeError, ValueError):
        return (1, str(question_id))


def build_sheet_report(question_reports: list, sheet_meta: dict = None) -> dict:
    """Roll per-question reports up into one report for the whole sheet.

    A sheet needs human review if ANY question does; `questions_needing_review`
    says which, so a reviewer opens those questions instead of re-marking
    everything."""
    ordered = sorted(question_reports, key=lambda r: _qsort_key(r["question_id"]))
    awarded = round(sum(r["total_awarded"] for r in ordered), 2)
    possible = round(sum(r["total_possible"] for r in ordered), 2)
    needing_review = [r["question_id"] for r in ordered if r["needs_human_review"]]
    return {
        "sheet": sheet_meta or {},
        "summary": {
            "total_awarded": awarded,
            "total_possible": possible,
            "percentage": round(100 * awarded / possible, 1) if possible else None,
            "questions_graded": len(ordered),
            "needs_human_review": bool(needing_review),
            "questions_needing_review": needing_review,
            "warnings": (sheet_meta or {}).get("warnings", []),
        },
        "questions": ordered,
    }


def print_sheet_summary(sheet_report: dict):
    s = sheet_report["summary"]
    print("=" * 64)
    pct = f" ({s['percentage']}%)" if s["percentage"] is not None else ""
    print(f"SHEET TOTAL: {s['total_awarded']} / {s['total_possible']}{pct}")
    if s["needs_human_review"]:
        print(f"Needs human review: questions {', '.join(s['questions_needing_review'])}")
    else:
        print("No review flags on any question")
    for w in s["warnings"]:
        print(f"  ! {w}")
    print("-" * 64)
    for q in sheet_report["questions"]:
        flag = "  [REVIEW]" if q["needs_human_review"] else ""
        print(f"  Q{q['question_id']:<3} {q['total_awarded']:>6} / {q['total_possible']:<6}"
              f" ocr={q.get('ocr_backend', '?')}{flag}")
    print("=" * 64)
