import sys
sys.path.append('c:/Users/soham/Downloads/agent ai exam - Copy/files(3)')
from PIL import Image
import llm_client
import pymupdf
import io

pdf_path = r'c:/Users/soham/Downloads/agent ai exam - Copy/second_ai/exam/Student_Pdf/Student_5.pdf'
doc = pymupdf.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

prompt = "Analyze this handwritten exam page. Identify each question being answered. Return a JSON list of objects, where each object has 'question_number' and 'bbox'. The 'bbox' should be the normalized coordinates [ymin, xmin, ymax, xmax] scaled between 0 and 1000 for the region containing the answer to that question."
response = llm_client.generate_text(prompt, image=image)
print(response)
