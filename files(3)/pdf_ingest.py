import io

import pymupdf
from PIL import Image

import config


def render_pdf_pages(pdf_path: str):
    """Render a PDF into a list of PIL images, one per page."""
    zoom = config.PDF_RENDER_DPI / 72
    doc = pymupdf.open(pdf_path)
    images = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(image)
    finally:
        doc.close()
    return images
