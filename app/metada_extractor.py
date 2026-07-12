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