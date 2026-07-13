from http.client import responses

from lib2to3.btm_utils import reduce_tree

import fitz, re, json
from datetime import datetime
from app.rag_chain import RAGChain

class MetadataExtractor:

    def __init__(self, rag_chain: RAGChain):
        """
        Recibe RAGChain ya instanciado para usar el LLM.
        No instancia nada propio — solo guarda la referencia.

        Input:  rag_chain → instancia ya creada de RAGChain
        Output: nada
        — self.rag_chain queda guardado
        """
        self.rag_chain = rag_chain

    def _is_empty(self, value) -> bool:
        """
        Verifica si un campo de metadata está vacío.
        Considera vacíos: None, "", " ", "none", "null", "unknown".
        Función interna usada en merge_metadata() y extract().

        Input:  value → cualquier valor de metadata
        Output: bool → True si está vacío o es placeholder inútil
        """
        if value is None:
            return True
        cleaned = str(value).strip().lower()
        return cleaned in {"", "none", "null", "unknown", "untitled", "sin título", "-"}

    def _parse_pdf_year(self, date_str:str) -> str:
        """
        Extrae el año del formato de fecha de PDF.
        El formato estándar es "D:YYYYMMDDHHmmSS".
        Función interna usada en extract_from_pdf().

        Input:  date_str → "D:20240115103200" o "" o None
        Output: str → "2024" o "" si no se puede extraer
        """
        if not date_str:
            return ""
        # El año está siempre en posiciones 2-6 si empieza con "D:"
        if date_str.startswith("D:") and len(date_str) >= 6:
            year = date_str[2:6]
            if year.isdigit() and 1900 <= int(year) <= 2100:
                return year
        # Fallback: busca 4 dígitos consecutivos que parezcan año
        match = re.search(r"(19|20)\d{2}", date_str)
        return match.group() if match else ""

    def _clean_field(self, value:str) -> str:
        """
        Limpia un campo de metadata eliminando caracteres
        extraños comunes en PDFs mal formados.
        Función interna usada en extract_from_pdf().

        Input:  value → string crudo de fitz.metadata
        Output: str → string limpio
        """
        if not value:
            return ""
        # Elimina caracteres de control y no imprimibles
        cleaned = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", value)
        # Elimina espacios múltiples
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def extract_from_pdf(self, pdf_path: str) -> dict:
        """
        Nivel 1 — extrae metadata embebida del PDF usando fitz.

        Input:  pdf_path → ruta completa del PDF en disco
        Output: dict con campos extraídos (vacíos si no existen)
                {
                  "title":    "Aprendizaje Automático",
                  "author":   "García, Juan",
                  "subject":  "Machine Learning",
                  "keywords": "regresión, clasificación",
                  "year":     "2024"
                }
        """
        empty = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "year": ""
        }
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            doc.close()
        except Exception as e:
            print(f"[MetadataExtractor] Error abriendo PDF: {e}")
            return empty
        return {
            "title": self._clean_field(meta.get("title", "")),
            "author": self._clean_field(meta.get("author", "")),
            "subject": self._clean_field(meta.get("subject", "")),
            "keywords": self._clean_field(meta.get("keywords", "")),
            "year": self._parse_pdf_year(
                meta.get("creationDate", "")
                or meta.get("modDate", "")
            )
        }

    def extract_from_filename(self, filename: str) -> dict:
        """
        Nivel 2 — infiere metadata del nombre del archivo.
        Solo llena campos que Nivel 1 dejó vacíos.

        Input:  filename → nombre del archivo sin ruta
                           ej. "apuntes_redes_2024.pdf"
        Output: dict con campos inferidos
                {
                  "title":    "apuntes redes",
                  "year":     "2024",
                  "keywords": "apuntes, redes"
                }
        """
        #Elimina la extension del archivo
        name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)

        # Extrae año si aparece (4 dígitos entre 1900-2099)
        year_match = re.search(r"(19|20)\d{2}", name)
        year = year_match.group() if year_match else ""

        #Elimina el año del nombre para no incluirlo en el título
        name_without_year = re.sub(r"(19|20)\d{2}", "", name)

        # Reemplaza separadores comunes por espacios
        name_clean = re.sub(r"[_\-\.]+", " ", name_without_year)
        name_clean = re.sub(r"\s+", " ", name_clean).strip().lower()

        #Las palabras del nombre son las keyword inferidas
        words = [word for word in name_clean.split() if len(word) > 2]
        keywords = ", ".join(words) if words else ""
        title = name_clean.title() if name_clean else ""
        return {
            "title": title,
            "author": "", #No se puede inferir el nombre de esta manera
            "subject": "", #No se puede inferir el nombre de esta manera
            "keywords": keywords,
            "year": year
        }

    def extract_with_llm(self, first_chunks: list[dict]) -> dict:
        """
        Nivel 3 — usa el LLM para inferir metadata.
        SOLO se activa si title Y subject están vacíos
        después de Niveles 1 y 2.

        Input:  first_chunks → primeros 3 chunks del documento
                               (output parcial de chunk_text)
        Output: dict con metadata inferida por el LLM en JSON
                {
                  "title":    "Sistema de Gestión de Redes",
                  "author":   "No identificado",
                  "subject":  "Redes de Computadoras",
                  "keywords": "TCP/IP, routing, switching"
                }
        """
        empty = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "year": ""
        }
        if not first_chunks:
            return empty

        # Construye texto de muestra con los primeros 3 chunks
        sample_text = "\n\n".join([
            chunk["text"][:500]   # primeros 500 chars de cada chunk
            for chunk in first_chunks[:3]
        ])
        prompt_chunks = [{
            "text": (
                f"Analiza el siguiente texto académico y extrae "
                f"la metadata en formato JSON con exactamente estas keys: "
                f"title, author, subject, keywords.\n\n"
                f"Reglas:\n"
                f"- title: título del documento o tema principal\n"
                f"- author: autor si aparece, sino 'No identificado'\n"
                f"- subject: área temática en 2-4 palabras\n"
                f"- keywords: 3-5 palabras clave separadas por coma\n"
                f"- Responde SOLO con el JSON, sin explicaciones\n\n"
                f"Texto:\n{sample_text}"
            ),
            "metadata": {"source": "llm_metadata_extraction"}
        }]

        try:
            response = self.rag_chain.generate_summary(prompt_chunks)
            # Limpia el response, el LLM a veces agrega ```json
            response_clean = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
            parsed = json.loads(response_clean)

            return {
                "title": self._clean_field(str(parsed.get("title", ""))),
                "author": self._clean_field(str(parsed.get("author", ""))),
                "subject": self._clean_field(str(parsed.get("subject", ""))),
                "keywords": self._clean_field(str(parsed.get("keywords", ""))),
                "year": ""
            }
        except json.JSONDecodeError:
            print("[MetadataExtractor] LLM no retornó JSON válido "
                  f"- response: {response[:100]}")
            return empty
        except Exception as e:
            print(f"[MetadataExtractor] Error en Nivel 3: {e}")
            return empty



    def merge_metadata(self,
                       pdf_meta: dict,
                       filename_meta: dict,
                       llm_meta: dict) -> dict:
        """
        Combina los tres niveles con prioridad Nivel 1 > 2 > 3.
        Un campo con valor en Nivel 1 nunca es sobreescrito.
        Agrega campo "metadata_source" indicando de dónde vino cada campo.

        Input:  pdf_meta      → output de extract_from_pdf()
                filename_meta → output de extract_from_filename()
                llm_meta      → output de extract_with_llm()
                                ({} si Nivel 3 no se activó)
        Output: dict final consolidado
                {
                  "title":           "Aprendizaje Automático",
                  "author":          "García, Juan",
                  "subject":         "Machine Learning",
                  "keywords":        "regresión, clasificación",
                  "year":            "2024",
                  "metadata_source": "pdf"  ← pdf | filename | llm | mixed
                }
        """

    def extract(self, pdf_path: str,
                filename: str,
                first_chunks: list[dict]) -> dict:
        """
        Función principal — orquesta los tres niveles en orden.
        Es la ÚNICA función que process() llama de este módulo.

        Input:  pdf_path     → ruta del PDF en disco
                filename     → nombre del archivo sin ruta
                first_chunks → primeros 3 chunks para Nivel 3
        Output: dict final de merge_metadata()
        """