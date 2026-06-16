from app.vector_store import VectorStore


class Dashboard:

    def __init__(self, vector_store: VectorStore):
        """
        Recibe el VectorStore ya instanciado.
        Mismo patrón que RAGChain — no crea una nueva conexión.

        Input:  vector_store → instancia ya creada de VectorStore
        Output: nada
        """
        self.vector_store = vector_store

    def get_global_stats(self) -> dict:
        """
        Retorna métricas globales para las tarjetas del dashboard.

        Input:  nada
        Output: dict
                {
                  "total_documents": 3,
                  "total_chunks":    124,
                  "total_pages":     45
                }
        """
        return self.vector_store.get_stats()

    def get_documents_table(self) -> list[dict]:
        """
        Retorna datos de documentos listos para st.dataframe().

        Input:  nada
        Output: lista de dicts
                [
                  {
                    "Documento":  "tesis.pdf",
                    "Chunks":     47,
                    "Páginas":    12
                  },
                  ...
                ]

        Nota: los nombres de las keys son los headers de la tabla,
              por eso van en español con mayúscula.
        """
        docs = self.vector_store.get_all_documents()
        return [
            {
                "Documento": doc["source"],
                "Chunks": doc["total_chunks"],
                "Páginas": doc["pages"]
            }
            for doc in docs
        ]

    def get_chunks_chart_data(self) -> dict:
        """
        Retorna datos formateados para st.bar_chart().

        Input:  nada
        Output: dict donde key=nombre del PDF, value=total de chunks
                {
                  "tesis.pdf":   47,
                  "paper_ml.pdf": 33,
                  "apuntes.pdf":  44
                }
        """
        docs = self.vector_store.get_all_documents()
        return {doc["source"]: doc["total_chunks"] for doc in docs}