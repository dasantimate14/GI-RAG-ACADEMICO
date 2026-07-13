import chromadb
from chromadb.config import Settings
import os
from app.embeddings import EmbeddingsManager
from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K_RESULTS
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
            "n_results": TOP_K_RESULTS,
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

    def get_document_embeddings(self, source: str) -> list[list[float]]:
        """
        Retorna todos los embeddings de chunks de un documento.

        Input:  source → nombre del archivo ("tesis.pdf")
        Output: list[list[float]] → un vector por chunk
                [[0.12, -0.45, ...], [0.33, 0.91, ...], ...]
        """
        result = self.collection.get(
            where={"source": source},
            include=["embeddings"]
        )
        embeddings = result.get("embeddings", [])
        if embeddings is None:
            return []
        return [e.tolist() if hasattr(e, "tolist") else e
                for e in embeddings]

    def get_all_sources(self) -> list[str]:
        """
        Retorna nombres únicos de todos los documentos indexados.
        Usada por MLClassifier.get_all_document_embeddings().

        Input:  nada
        Output: list[str] → ["tesis.pdf", "paper.pdf", "apuntes.pdf"]
        """
        result    = self.collection.get(include=["metadatas"])
        metadatas = result["metadatas"]
        sources   = list({m["source"] for m in metadatas if "source" in m})
        return sources
