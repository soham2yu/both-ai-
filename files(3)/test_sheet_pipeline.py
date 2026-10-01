"""
Tests for the staged pipeline. Builds its own fixtures in a temp dir, so it
needs nothing but the code and a Tesseract install.

Ollama is NOT assumed to be running: the escalation logic is exercised by
patching llm_client, and everything else is checked in its no-LLM
degraded mode.

    python3 test_sheet_pipeline.py
"""

import io
import json
import os
import random
import tempfile

import pymupdf
from PIL import Image, ImageFilter

import config
import llm_client
import ocr
import segmenter
import sheet_pipeline as sp

PASSED = []


def check(name, condition, detail=""):
    assert condition, f"FAILED: {name} {detail}"
    PASSED.append(name)
    print(f"  ok  {name}")


KEY_MD = """# Answer Key
**Q1 [6 marks]:** Newton's third law states that for every action there is an equal and opposite reaction. Forces act on two different bodies.

**Q2 [8 marks]:** Photosynthesis converts light energy into chemical energy. It takes place in the chloroplasts of plant cells. Plants absorb carbon dioxide and water to produce glucose and oxygen.

**Q3 [4 marks]:** A rocket moves forward by expelling exhaust gas backward. This follows Newton's third law.
"""

PAPER_MD = """Q1. State and explain Newton's third law. [6 marks]
Q2. Describe photosynthesis. [8 marks]
Q3. Explain how a rocket moves. [4 marks]
"""

ANSWERS = {
    1: ("Newton's third law says that for every action there is an equal and opposite reaction. "
        "The two forces act on two different bodies. I also enjoy studying physics a lot."),
    2: ("Photosynthesis converts light energy into chemical energy. It takes place in the chloroplasts "
        "of plant cells. Plants absorb carbon dioxide and water and produce glucose and oxygen."),
    3: ("A rocket moves forward by expelling exhaust gas backward. This follows Newton's third law "
        "of motion."),
}


def _digital_page(text):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(50, 60, 545, 700), text, fontsize=16)
    return doc


def _rasterise(text, dpi=200):
    doc = _digital_page(text)
    pix = doc[0].get_pixmap(dpi=dpi)
    doc.close()
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _degrade(img):
    random.seed(7)
    img = img.filter(ImageFilter.GaussianBlur(3.2)).rotate(4, fillcolor=(255, 255, 255))
    img = img.resize((img.width // 3, img.height // 3)).resize(img.size)
    px = img.load()
    for _ in range(img.width * img.height // 6):
        x, y = random.randrange(img.width), random.randrange(img.height)
        v = random.randrange(256)
        px[x, y] = (v, v, v)
    return img


def build_student_pdf(path, pages):
    """pages: list of ('text', str) | ('image', PIL.Image)"""
    doc = pymupdf.open()
    for kind, payload in pages:
        page = doc.new_page(width=595, height=842)
        if kind == "text":
            page.insert_textbox(pymupdf.Rect(50, 60, 545, 700), payload, fontsize=16)
        else:
            buf = io.BytesIO()
            payload.save(buf, format="PNG")
            page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()


def test_segmenter():
    print("segmenter")
    s = segmenter.segment_sentences("The force acts on two dif-\nferent bodies and is equal\nand opposite. Next point")
    check("joins wrapped lines + de-hyphenates", s[0] == "The force acts on two different bodies and is equal and opposite.", s)
    s = segmenter.segment_sentences("Steps:\n1. First step\n2. Second step")
    check("list items are separate, markers stripped", s == ["Steps:", "First step", "Second step"], s)
    check("empty input", segmenter.segment_sentences("  \n ") == [])


def test_ocr_escalation():
    print("ocr escalation (Ollama mocked)")
    img = Image.new("RGB", (100, 100), "white")
    real_t, real_g = ocr.tesseract_ocr, llm_client.generate_text
    try:
        # confident Tesseract -> vision must NOT be called
        ocr.tesseract_ocr = lambda i: {"text": "clear text", "confidence": 0.9}
        llm_client.generate_text = lambda *a, **k: (_ for _ in ()).throw(AssertionError("vision called needlessly"))
        r = ocr.ocr_image(img)
        check("confident Tesseract skips vision", r["backend"] == "tesseract" and r["text"] == "clear text")

        # low confidence + Ollama answers -> vision result wins
        ocr.tesseract_ocr = lambda i: {"text": "g@rbl3d", "confidence": 0.31}
        llm_client.generate_text = lambda *a, **k: "Here is the transcription:\nreal handwritten text"
        r = ocr.ocr_image(img)
        check("low confidence escalates to vision", r["backend"] == "ollama_vision", r)
        check("vision preamble stripped", r["text"] == "real handwritten text", r["text"])
        check("vision confidence is the configured flat value", r["confidence"] == config.OCR_OLLAMA_VISION_CONFIDENCE)

        # low confidence + Ollama down -> keep Tesseract text, cap confidence
        llm_client.generate_text = lambda *a, **k: None
        r = ocr.ocr_image(img)
        check("Ollama unavailable keeps Tesseract text", r["backend"] == "tesseract_low_confidence" and r["text"] == "g@rbl3d")
        check("confidence capped so it gets flagged", r["confidence"] <= config.OCR_LOW_CONFIDENCE_CAP)

        # nothing usable at all
        ocr.tesseract_ocr = lambda i: {"text": "", "confidence": 0.0, "error": "tesseract_not_installed"}
        r = ocr.ocr_image(img)
        check("no engine at all degrades, does not raise", r["backend"] == "unavailable" and r["confidence"] == 0.0)
    finally:
        ocr.tesseract_ocr, llm_client.generate_text = real_t, real_g


def test_pipeline(tmp):
    print("full staged pipeline")
    key, paper = os.path.join(tmp, "key.md"), os.path.join(tmp, "paper.md")
    open(key, "w").write(KEY_MD)
    open(paper, "w").write(PAPER_MD)

    sheet = os.path.join(tmp, "sheet.pdf")
    build_student_pdf(sheet, [
        ("text", ANSWERS[1]),                       # page 1: machine-readable text layer
        ("image", _rasterise(ANSWERS[2])),          # page 2: clean scan
        ("image", _degrade(_rasterise(ANSWERS[3]))),  # page 3: badly degraded scan
    ])
    out = os.path.join(tmp, "work")

    r = sp.stage_extract_key(key, out, paper)
    check("answer key parsed", r["questions"] == 3 and r["notes"] == [], r)
    ak = json.load(open(os.path.join(out, "answer_key.json")))
    check("answer_key.json shape {que_num: {answer, marks}}", ak["2"]["marks"] == 8.0 and "chloroplasts" in ak["2"]["answer"])

    r = sp.stage_decompose(out)
    rub = json.load(open(os.path.join(out, "rubric.json")))
    check("rubric.json written per question", set(rub) == {"1", "2", "3"} and all(v["points"] for v in rub.values()), r)
    for n, v in rub.items():
        check(f"Q{n} rubric marks sum to total", abs(sum(p["marks"] for p in v["points"]) - v["total_marks"]) < 0.011)

    r = sp.stage_extract_student(sheet, out)
    back = r["backends"]
    check("page 1 used the text layer (no OCR)", back["1"] == "text_layer", back)
    check("page 2 (clean scan) used Tesseract", back["2"] == "tesseract", back)
    check("page 3 (degraded) did not silently pass as confident",
          back["3"] in ("tesseract_low_confidence", "ollama_vision", "unavailable"), back)

    report = sp.stage_grade(out, "sheet.pdf")
    s = report["summary"]
    qs = {q["question_id"]: q for q in report["questions"]}
    check("report covers every question", set(qs) == {"1", "2", "3"})
    check("sheet total is the sum of questions",
          abs(s["total_awarded"] - sum(q["total_awarded"] for q in qs.values())) < 0.011)
    check("sheet possible = 6+8+4", s["total_possible"] == 18.0, s)
    check("question paper text carried into report", qs["1"]["question_text"].startswith("State and explain"))
    check("Q2 (clean scan of the key text) scores highly", qs["2"]["total_awarded"] >= 6.0, qs["2"]["total_awarded"])
    check("Q1 padding sentence flagged as extra content",
          any("enjoy studying physics" in e["sentence"] for e in qs["1"]["extra_content_flags"]))
    check("degraded page 3 is flagged for review", qs["3"]["needs_human_review"], qs["3"]["review_reasons"])
    check("sheet lists which questions need review", "3" in s["questions_needing_review"], s)
    check("report.json written", os.path.exists(os.path.join(out, "report.json")))
    print(f"      totals: {s['total_awarded']}/{s['total_possible']}  review: {s['questions_needing_review']}")
    return sheet, out


def test_failure_modes(tmp):
    print("failure modes")
    # marks missing -> decompose must refuse with an actionable message
    bad = os.path.join(tmp, "nomarks")
    os.makedirs(bad)
    open(os.path.join(tmp, "nomarks.md"), "w").write("Q1: Some answer without marks.\nQ2: Another answer.")
    sp.stage_extract_key(os.path.join(tmp, "nomarks.md"), bad)
    try:
        sp.stage_decompose(bad)
        check("missing marks rejected", False)
    except ValueError as e:
        check("missing marks rejected with actionable message", "Marks missing" in str(e) and "answer_key.json" in str(e), str(e))

    # sheet shorter than the paper -> unanswered question, warned + flagged
    out = os.path.join(tmp, "short")
    open(os.path.join(tmp, "key2.md"), "w").write(KEY_MD)
    sp.stage_extract_key(os.path.join(tmp, "key2.md"), out)
    sp.stage_decompose(out)
    short = os.path.join(tmp, "short.pdf")
    build_student_pdf(short, [("text", ANSWERS[1]), ("text", ANSWERS[2])])
    n = sp.stage_extract_student(short, out)["notes"]
    check("page/question mismatch warned at extraction", any("3" in x for x in n), n)
    rep = sp.stage_grade(out)
    q3 = [q for q in rep["questions"] if q["question_id"] == "3"][0]
    check("unanswered question scores 0 and is flagged", q3["total_awarded"] == 0 and q3["needs_human_review"])
    check("warning surfaced in sheet summary", any("Q3" in w for w in rep["summary"]["warnings"]), rep["summary"]["warnings"])

    # non-PDF student sheet
    try:
        sp.stage_extract_student(os.path.join(tmp, "key2.md"), out)
        check("non-PDF sheet rejected", False)
    except ValueError:
        check("non-PDF sheet rejected", True)

    # grading before earlier stages
    try:
        sp.stage_grade(os.path.join(tmp, "does_not_exist"))
        check("missing artifacts reported", False)
    except FileNotFoundError as e:
        check("missing artifacts reported clearly", "run the earlier stage" in str(e))


def test_handwritten_image_sheet(tmp):
    print("handwritten image sheet")
    key = os.path.join(tmp, "key_handwritten.md")
    paper = os.path.join(tmp, "paper_handwritten.md")
    open(key, "w").write(KEY_MD)
    open(paper, "w").write(PAPER_MD)

    sheet = os.path.join(tmp, "student_handwritten.png")
    _rasterise(ANSWERS[1] + "\n\n" + ANSWERS[2] + "\n\n" + ANSWERS[3]).save(sheet)

    out = os.path.join(tmp, "handwritten_work")
    report = sp.evaluate_handwritten_sheet(key, paper, sheet, out)
    check("handwritten image sheet is graded end-to-end", "summary" in report and report["summary"]["total_possible"] == 18.0, report)
    check("image-based sheet writes report artifact", os.path.exists(os.path.join(out, "report.json")))


def test_cli(tmp, sheet):
    print("CLI")
    out = os.path.join(tmp, "cli_work")
    rc = sp.main(["run", "--key", os.path.join(tmp, "key.md"), "--paper", os.path.join(tmp, "paper.md"),
                  "--sheet", sheet, "--out", out])
    check("`run` exits 0 and writes all artifacts",
          rc == 0 and all(os.path.exists(os.path.join(out, f)) for f in
                          ("question_paper.json", "answer_key.json", "rubric.json", "student_text.json", "report.json")))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_segmenter()
        test_ocr_escalation()
        sheet, _ = test_pipeline(tmp)
        test_failure_modes(tmp)
        test_cli(tmp, sheet)
    print(f"\nAll {len(PASSED)} checks passed")
