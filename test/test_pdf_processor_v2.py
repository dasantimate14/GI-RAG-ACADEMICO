import sys
import os

# Asegurar que la raíz del proyecto está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pdf_processor import PDFProcessor

# Prueba 1 — _count_words()
processor = PDFProcessor()
pages = [{"page": 1, "text": "Hola mundo como estas"},
         {"page": 2, "text": "Bien gracias y tu"}]
resultado = processor._count_words(pages)
assert resultado == 8, f"Esperado 8, obtuvo {resultado} (Nota: El texto de ejemplo tiene 8 palabras en total)"
print("OK: _count_words OK")

# Prueba 2 — process() sin metadata_extractor retorna dict
processor = PDFProcessor()
# Usa un PDF real de ./data/uploads/
pdfs = [f for f in os.listdir("./data/uploads") if f.endswith(".pdf")]
if not pdfs:
    print("WARNING: Pon un PDF en ./data/uploads/ para esta prueba")
else:
    # Simula el objeto de Streamlit
    class FakeFile:
        name = pdfs[0]
        def __init__(self):
            with open(f"./data/uploads/{self.name}", "rb") as f:
                self.content = f.read()
        def getbuffer(self):
            return self.content
    resultado = processor.process(FakeFile())
    assert isinstance(resultado, dict), "process() debe retornar dict"
    assert "chunks"   in resultado, "Falta key 'chunks'"
    assert "metadata" in resultado, "Falta key 'metadata'"
    assert "stats"    in resultado, "Falta key 'stats'"
    assert isinstance(resultado["chunks"], list), "chunks debe ser list"
    assert resultado["stats"]["total_chunks"] == len(resultado["chunks"])
    assert resultado["stats"]["total_pages"] > 0
    assert resultado["stats"]["total_words"] > 0
    assert resultado["metadata"]["source"] == pdfs[0]
    print(f"OK: process() retorna dict OK")
    print(f"   chunks:      {resultado['stats']['total_chunks']}")
    print(f"   paginas:     {resultado['stats']['total_pages']}")
    print(f"   palabras:    {resultado['stats']['total_words']}")
    print(f"   metadata:    {resultado['metadata']}")

    # Prueba 3 — stats son consistentes
    assert resultado["stats"]["total_chunks"] > 0
    assert resultado["stats"]["total_pages"]  > 0
    assert resultado["stats"]["total_words"]  > 0
    assert "upload_date" in resultado["stats"]
    print("OK: stats consistentes OK")
