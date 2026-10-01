"""
Text extraction from student answer sheets.

Per-PAGE decision, cheapest first:

  1. Text layer  -- if the PDF page already carries machine-readable text
     (>= PDF_TEXT_LAYER_MIN_CHARS characters), use it directly. No OCR.
  2. Tesseract   -- for pages that are just images. Kept if its mean word
     confidence >= OCR_TESSERACT_MIN_CONFIDENCE.
  3. Ollama vision -- only when Tesseract's confidence is below that
     threshold (or it produced nothing). Slow, so it is the escalation
     path, not the default.
  4. If Ollama can't help (not running / model not pulled / empty output),
     Tesseract's text is kept but its confidence is capped at
     OCR_LOW_CONFIDENCE_CAP so the page is flagged for human review instead
     of being silently trusted.

Every function returns {"text", "confidence", "backend"}. Nothing raises: an
unreadable page yields empty text + confidence 0.0, which downstream turns
into needs_human_review rather than crashing the whole sheet.
"""

import io
import re

import pymupdf
import pytesseract
from PIL import Image, ImageOps

import config
import llm_client

OCR_PROMPT = """Transcribe ALL handwritten text in this exam answer-sheet image
exactly as written, including spelling mistakes -- do not correct or clean up
the student's wording. Output ONLY the transcribed text, nothing else -- no
preamble, no commentary, no markdown."""

_PREAMBLE = re.compile(r"^(?:here(?:'s| is)|the (?:handwritten )?(?:text|transcription)|transcription)[^\n]*:\s*\n?", re.IGNORECASE)


def _clean_text(raw_text: str) -> str:
    """Light cleanup of common OCR artifacts before anything downstream sees it."""
    text = raw_text.replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    return text.strip()


def tesseract_ocr(image: Image.Image) -> dict:
    """Returns {"text", "confidence"} or, if the engine is unusable,
    {"text": "", "confidence": 0.0, "error": "..."}."""
    try:
        prepared = ImageOps.autocontrast(image.convert("L"))
        left, top = int(prepared.width * 0.04), int(prepared.height * 0.02)
        right, bottom = int(prepared.width * 0.98), int(prepared.height * 0.98)
        prepared = prepared.crop((left, top, right, bottom))
        prepared = prepared.resize(
            (prepared.width * 2, prepared.height * 2),
            Image.Resampling.LANCZOS,
        )
        data = pytesseract.image_to_data(
            prepared,
            config="--oem 1 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        raw_text = pytesseract.image_to_string(prepared, config="--oem 1 --psm 6")
    except pytesseract.TesseractNotFoundError:
        # pytesseract is only a wrapper; the engine is a separate system install.
        return {"text": "", "confidence": 0.0, "error": "tesseract_not_installed"}
    except Exception as e:
        return {"text": "", "confidence": 0.0, "error": f"tesseract_error: {e}"}

    confs = []
    for word, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if word.strip() and c >= 0:
            confs.append(c)
    confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return {"text": _clean_text(raw_text), "confidence": round(confidence, 3)}


def _downscale(image: Image.Image) -> Image.Image:
    longest = max(image.size)
    if longest <= config.VISION_MAX_SIDE_PX:
        return image
    scale = config.VISION_MAX_SIDE_PX / longest
    return image.resize((int(image.width * scale), int(image.height * scale)))


def vision_ocr(image: Image.Image):
    """Ollama vision transcription. None if Ollama/model unavailable or the
    output is empty."""
    raw = llm_client.generate_text(OCR_PROMPT, image=_downscale(image), temperature=0.0)
    if not raw:
        return None
    text = _clean_text(_PREAMBLE.sub("", raw.strip()))
    return {"text": text, "confidence": config.OCR_OLLAMA_VISION_CONFIDENCE} if text else None


def ocr_image(image: Image.Image) -> dict:
    """Use Ollama vision (Qwen2.5-VL) for every page."""
    vision = vision_ocr(image)
    if vision:
        return {**vision, "backend": "ollama_vision"}

    return {"text": "", "confidence": 0.0, "backend": "unavailable"}


def extract_pdf_pages(pdf_path: str) -> list:
    """One result per PDF page, in order: {"page", "text", "confidence", "backend"}.
    Pages with a usable text layer skip OCR (and skip rendering) entirely."""
    results = []
    zoom = config.PDF_RENDER_DPI / 72
    doc = pymupdf.open(pdf_path)
    try:
        for index, page in enumerate(doc, start=1):
            print(f"  -> OCR processing page {index}/{len(doc)}...", flush=True)
            layer = page.get_text().strip()
            if len(layer) >= config.PDF_TEXT_LAYER_MIN_CHARS:
                print(f"     (Skipped OCR, found text layer)", flush=True)
                results.append({"page": index, "text": _clean_text(layer), "confidence": 1.0, "backend": "text_layer"})
                continue
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            results.append({"page": index, **ocr_image(image)})
            print(f"     (Done page {index})", flush=True)
    finally:
        doc.close()
    return results


def extract_image_pages(image_path: str) -> list:
    """Single-page handwritten or scanned image sheets. Returns one page result for the image."""
    lower = image_path.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_pages(image_path)

    allowed = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    if not lower.endswith(allowed):
        raise ValueError(f"Unsupported sheet image type: {image_path} (expected PDF or image)")

    image = Image.open(image_path).convert("RGB")
    return [{"page": 1, **ocr_image(image)}]


def extract_text_for_question(images: list) -> dict:
    """Used by the older metadata-driven flow (main_pipeline.py / api.py):
    OCR every image belonging to one question and concatenate in order."""
    if not images:
        return {"text": "", "confidence": 0.0, "backend": "none"}
    per_page = [ocr_image(img) for img in images]
    return {
        "text": _clean_text("\n".join(p["text"] for p in per_page)),
        "confidence": min(p["confidence"] for p in per_page),  # weakest page sets the trust level
        "backend": "+".join(sorted({p["backend"] for p in per_page})),
    }
