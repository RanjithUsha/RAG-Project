# Load PDF → split → embed → persist in Chroma (CLI helper; UI uses app.py)

from pathlib import Path

from dotenv import load_dotenv

from rag_utils import index_pdf_file

load_dotenv()

pdf_path = (
    Path(__file__).parent
    / "Documnet Loaders"
    / "OReilly.Fundamentals.of.Deep.Learning.2017.5.pdf"
)

count = index_pdf_file(pdf_path)
print(f"Indexed {count} chunks into chroma_db.")
