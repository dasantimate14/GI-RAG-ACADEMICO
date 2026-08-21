import fitz
import os
import re
from datetime import datetime, timezone
from app.metada_extractor import MetadataExtractor
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    UPLOAD_PATH

)

class PDFProcessor:

    def __init__(self):
        """
        Inicializa el procesador.
        Recibe MetadataExtractor ya instanciado
        Crea la carpeta UPLOADS_PATH si no existe.

        Input:  nada
        Output: nada
        — self.metadata_extractor:   MetadataExtractor
        """
        if not os.path.exists(UPLOAD_PATH):
            os.makedirs(UPLOAD_PATH, exist_ok=True)

    def save_pdf(self, uploaded_file) -> str:
        """
        Guarda el archivo PDF subido desde Streamlit en disco.

        Input: uploaded_file → objeto de st.file_uploader()
        Output: str → ruta completa donde se guardó el archivo
                      ej."./data/uploads/tesis.pdf"
        """
        file_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if os.path.exists(file_path):
            return str(file_path)
        else:
            return ""


    def extract_text(self, pdf_path: str) -> list[dict]:
        """
        Abre el PDF y extrae el texto de cada página.

        Input: pdf_path → ruta del PDF en disco
        Output: lista de dicts, uno por página
                [
                  {"page": 1, "text": "contenido de la página..."},
                  {"page": 2, "text": "contenido de la página..."},
                ]
        """
        pages_content = []
        file = fitz.open(pdf_path)
        for page_num in range(len(file)):
            page = file.load_page(page_num)
            text = page.get_text()
            pages_content.append({"page": page_num + 1, "text": text})
            print(f"---Page {page_num + 1}---")
            print(text)
        file.close()
        return pages_content


    def clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído eliminando ruido.
        (saltos de línea múltiples, espacios extra, caracteres raros)

        Input:  text → string crudo extraído del PDF
        Output: str  → texto limpio
        """

        if not text:
            return ""
        # Normalizar saltos de línea
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Unir palabras separadas por guiones al final de línea
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Convertir saltos de línea dentro de párrafos en espacios
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Mantener separación entre párrafos
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        # Eliminar espacios múltiples
        text = re.sub(r"[ \t]+", " ", text)
        # Eliminar caracteres no imprimibles
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
        # Eliminar números de página aislados
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()


    def chunk_text(self, pages: list[dict], source: str) -> list[dict]:
        """
        Divide el texto de todas las páginas en chunks con overlap.
        Usa CHUNK_SIZE y CHUNK_OVERLAP de config.py.

        Input:  pages  → output de extract_text()
                source → nombre del archivo PDF (para metadata)
        Output: lista de dicts, uno por chunk
                [
                  {
                    "text": "fragmento de texto...",
                    "metadata": {
                      "source":   "tesis.pdf",
                      "page":     1,
                      "chunk_id": 0
                    }
                  },
                  ...
                ]
        """
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        chunk_id = 0

        for page in pages:
            cleaned_text = page["text"]
            # Split text into paragraphs, then into individual sentences
            paragraphs = cleaned_text.split('\n')

            for paragraph in paragraphs:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]

                for sentence in sentences:
                    sentence_size = len(sentence)

                    # Prevent rare, massive lines from breaking the chunk limits
                    if sentence_size > CHUNK_SIZE:
                        sentence = sentence[:CHUNK_SIZE]
                        sentence_size = len(sentence)

                    # Check if adding this sentence triggers a chunk overflow
                    if current_size + sentence_size > CHUNK_SIZE and current_chunk_sentences:
                        # Compile the full text block for this chunk
                        chunk_text_content = " ".join([item["text"] for item in current_chunk_sentences])
                        chunk_pages = sorted(list(set(item["page"] for item in current_chunk_sentences)))

                        chunks.append({
                            "text": chunk_text_content,
                            "metadata": {
                                "source": source,
                                "chunk_id": chunk_id,
                                "page_start": chunk_pages[0],
                                "page_end": chunk_pages[-1],
                                "pages": chunk_pages
                            }
                        })
                        chunk_id += 1

                        # Safely build the overlap using sentences from the end of the current chunk
                        overlap_sentences = []
                        overlap_size = 0
                        for item in reversed(current_chunk_sentences):
                            if overlap_size + len(item["text"]) <= CHUNK_OVERLAP:
                                overlap_sentences.insert(0, item)
                                overlap_size += len(item["text"])
                            else:
                                break

                        current_chunk_sentences = overlap_sentences
                        current_size = overlap_size

                    # Safely track each sentence paired with its source page
                    current_chunk_sentences.append({"text": sentence, "page": page["page"]})
                    current_size += sentence_size

        # Flush out any remaining text into a final chunk
        if current_chunk_sentences:
            chunk_text_content = " ".join([item["text"] for item in current_chunk_sentences])
            chunk_pages = sorted(list(set(item["page"] for item in current_chunk_sentences)))
            chunks.append({
                "text": chunk_text_content,
                "metadata": {
                    "source": source,
                    "chunk_id": chunk_id,
                    "page_start": chunk_pages[0],
                    "page_end": chunk_pages[-1],
                    "pages": chunk_pages
                }
            })

        return chunks

    def _count_words(self, pages: list[dict]) -> int:
        """
        Cuenta palabras totales en todas las páginas.
        Función interna usada en process().

        Input:  pages → output de extract_text()
        Output: int → total de palabras en el documento
        """
        total = 0
        for page in pages:
            total += len(page["text"].split())
        return total

    def process(self, uploaded_file, metadata_extractor: MetadataExtractor = None) -> dict:
        """
        Función principal — orquesta el pipeline de un PDF.
        Llama internamente a: save_pdf → extract_text → clean_text → chunk_text → MetadataExtractor.extract() → _count_words()

        Input:  uploaded_file → objeto de st.file_uploader()
        Output: Diccionario completo
            {
              "chunks":   [...],
              "metadata": {...},   ← NUEVO: output de MetadataExtractor
              "stats": {           ← NUEVO
                "total_chunks": 47,
                "total_pages":  12,
                "total_words":  3420,
                "upload_date":  "2024-01-15T10:32:00"
              }
            }

        Es la ÚNICA función que main.py necesita llamar de este módulo.
        """
        file_path = self.save_pdf(uploaded_file)
        if not file_path:
            raise ValueError(f"No se pudo guardar el archivo {uploaded_file}")
        pages_content = self.extract_text(file_path)
        for page in pages_content:
            page["text"] = self.clean_text(page["text"])
        chunks = self.chunk_text(pages_content, uploaded_file.name)
        if metadata_extractor is not None:
            metadata = metadata_extractor.extract(
                pdf_path=file_path,
                filename=uploaded_file.name,
                first_chunks=chunks[:3]
            )
        else:
            metadata = {
                "source": uploaded_file.name,
                "title": "",
                "author": "",
                "subject": "",
                "keywords": "",
                "year": "",
                "metadata_source": "none"
            }
        stats = {
            "total_chunks": len(chunks),
            "total_pages": len(pages_content),
            "total_words": self._count_words(pages_content),
            "upload_date": datetime.now(timezone.utc).isoformat()
        }
        metadata["source"] = uploaded_file.name
        return {"chunks": chunks, "metadata": metadata, "stats": stats}