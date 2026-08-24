import chromadb, os, re
from chromadb.config import Settings
from rank_bm25 import BM25Okapi #Implementación de BM25 con suavizado Okapi BM25
from app.embeddings import EmbeddingsManager
from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K_SEMANTIC,
    TOP_K_KEYWORD
)

class VectorStore:

    def __init__(self):
        """
        Conecta a ChromaDB persistente y obtiene/crea la colección.
        Usa CHROMA_PATH y COLLECTION_NAME de config.py.
        Instancia internamente un EmbeddingsManager.

        Input: nada
        Output: nada
        — self.client, self.collection y self.embedder quedan listos
        """
        self.client = chromadb.PersistentClient(CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.embedder = EmbeddingsManager()
        self._bm25_index = None
        self._bm25_corpus = []

    def _build_bm25_index(self) -> None:
        """
        Construye el índice BM25 con todos los chunks actualmente
        en ChromaDB. Se llama automáticamente (lazy) antes de la
        primera búsqueda por keyword, y se invalida cuando se
        agregan o eliminan documentos.

        Input:  nada
        Output: nada — modifica self._bm25_index y self._bm25_corpus
        """
        result = self.collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not docs:
            self._bm25_index = None
            self._bm25_corpus = []
            return

        # Tokeniza cada chunk para BM25
        tokenized = [
            re.sub(r'[^\w\s]', '', doc.lower()).split()
            for doc in docs
        ]

        self._bm25_index = BM25Okapi(tokenized)

        # Guarda el corpus original para retornar texto real en los resultados
        self._bm25_corpus = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(docs, metadatas)
        ]

    def search_by_keyword(self, query: str, filter_source: str = None, n_results:int = None) -> list[dict]:
        """
        Búsqueda léxica por palabras clave usando BM25Okapi.
        Complementa search() semántico para términos técnicos
        muy específicos que el embedding no captura bien.

        Ejemplo donde BM25 supera a semántica:
          query: "BM25Okapi hyperparameter k1"
          → embedding no sabe qué es BM25Okapi
          → BM25 encuentra exactamente ese término en los chunks

        Input:  query         → pregunta del usuario en texto plano
                filter_source → nombre de PDF para filtrar (opcional)
                n_results     → cuántos chunks retornar
        Output: list[dict] → chunks ordenados por relevancia BM25
                cada dict: {
                  "text":     str,
                  "metadata": dict,
                  "distance": float  ← score BM25 sin normalizar
                                       (se normaliza en rag_chain)
                }
        """
        k_results = n_results or TOP_K_KEYWORD

        #Construye el índice si no existe o fue invalidado
        if self._bm25_index is None:
            self._build_bm25_index()

        if self._bm25_index is None:
            return []

        #Tokeniza la query igual que el corpus
        tokens = re.sub(r'[^\w\s]', '', query.lower()).split()
        if not tokens:
            return []

        #Obtiene scores BM25 para todos los chunks
        scores = self._bm25_index.get_scores(tokens)

        # Empareja scores con chunks y ordena
        scored_chunks = sorted(
            zip(scores, self._bm25_corpus),
            key=lambda x: x[0],
            reverse=True
        )

        results = []
        for score, chunk in scored_chunks:
            # Filtra por documento si se especificó
            if filter_source:
                if chunk["metadata"].get("source") != filter_source:
                    continue

            # Solo incluye chunks con coincidencias reales
            if score <= 0:
                break   # sorted descendente → si score=0, los siguientes también

            results.append({
                "text":     chunk["text"],
                "metadata": chunk["metadata"],
                "distance": float(score)   # float() convierte numpy → Python
            })

            if len(results) >= k_results:
                break

        return results


    def add_documents(self, chunks: list[dict]) -> int:
        """
        Indexa una lista de chunks en ChromaDB.
        Genera embeddings internamente usando self.embedder.

        Input:  chunks → output de PDFProcessor.process()
                         [{"text": "...", "metadata": {...}}, ...]
        Output: int → número de chunks indexados exitosamente

        Internamente:
          1. Extrae los textos de los chunks
          2. Genera embeddings con self.embedder.generate()
          3. Construye IDs únicos para cada chunk
          4. Llama a self.collection.add()
        """
        if not chunks:
            return 0
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_id']}" for chunk in chunks]

        for metadata in metadatas:
            metadata["pages"] = str(metadata["pages"])

        embeddings = self.embedder.generate(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        self._bm25_index = None
        return len(chunks)


    def search(self, query: str, filter_source: str = None) -> list[dict]:
        """
        Busca los chunks más relevantes para una query.
        Usa TOP_K_RESULTS de config.py.

        Input:  query         → pregunta del usuario en texto plano
                filter_source → (opcional) nombre de PDF para buscar
                                solo dentro de ese documento
        Output: lista de dicts con los chunks más relevantes
                [
                  {
                    "text":     "contenido del chunk...",
                    "metadata": {"source": "tesis.pdf", "page": 3},
                    "distance": 0.23    ← qué tan similar es (menor = mejor)
                  },
                  ...
                ]
        """
        query_embeddings = self.embedder.generate_one(query)
        query_params = {
            "query_embeddings": [query_embeddings],
            "n_results": TOP_K_SEMANTIC,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter_source:
            query_params["where"] = {"source": filter_source}

        results = self.collection.query(**query_params)

        chunks = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for document, metadata, distance in zip(
                documents,
                metadatas,
                distances
        ):
            if "pages" in metadata and isinstance(metadata["pages"], str):
                try:
                    metadata["pages"] = eval(metadata["pages"])
                except:
                    pass
            chunks.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })
        return chunks


    def document_exists(self, source: str) -> bool:
        """
        Verifica si un PDF ya fue indexado anteriormente.
        Evita indexar el mismo documento dos veces.

        Input:  source → nombre del archivo PDF ("tesis.pdf")
        Output: bool → True si ya existe, False si no
        """
        results = self.collection.get(
            where = {"source": source},
            limit = 1
        )
        if results["ids"]:
            return True
        else:
            return False

    def delete_document(self, source: str) -> bool:
        """
        Elimina todos los chunks de un PDF específico de ChromaDB.

        Input:  source → nombre del archivo PDF a eliminar
        Output: bool → True si se eliminó, False si no existía
        """
        results = self.collection.get(
            where={"source": source},
            include=[]
        )
        ids = results.get("ids", [])
        if not ids:
            return False

        self.collection.delete(ids=ids)
        self._bm25_index = None
        return True

    def get_all_documents(self) -> list[dict]:
        """
        Retorna metadata de todos los documentos indexados.
        Usado principalmente por dashboard.py.

        Input:  nada
        Output: lista de dicts, uno por documento único
                [
                  {
                    "source":       "tesis.pdf",
                    "total_chunks": 47,
                    "pages":        12
                  },
                  ...
                ]
        """
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        if not metadatas:
            return []

        docs = {}
        for meta in metadatas:
            if not meta or "source" not in meta:
                continue
            source = meta["source"]
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "total_chunks": 0,
                    "pages_set": set()
                }
            docs[source]["total_chunks"] += 1

            pages_val = meta.get("pages", "[]")
            if isinstance(pages_val, str):
                try:
                    parsed_pages = eval(pages_val)
                    if isinstance(parsed_pages, list):
                        docs[source]["pages_set"].update(parsed_pages)
                except Exception:
                    pass
            elif isinstance(pages_val, list):
                docs[source]["pages_set"].update(pages_val)
            elif isinstance(pages_val, int):
                docs[source]["pages_set"].add(pages_val)

        result = []
        for source, info in docs.items():
            max_page = max(info["pages_set"]) if info["pages_set"] else 0
            result.append({
                "source": source,
                "total_chunks": info["total_chunks"],
                "pages": max_page
            })
        return result

    def get_stats(self) -> dict:
        """
        Retorna estadísticas globales de la colección.
        Usado por dashboard.py.

        Input:  nada
        Output: dict con métricas globales
                {
                  "total_documents": 3,
                  "total_chunks":    124,
                  "total_pages":     45
                }
        """
        docs_metadata = self.get_all_documents()
        total_documents = len(docs_metadata)
        total_chunks = sum(doc["total_chunks"] for doc in docs_metadata)
        total_pages = sum(doc["pages"] for doc in docs_metadata)

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_pages": total_pages
        }
