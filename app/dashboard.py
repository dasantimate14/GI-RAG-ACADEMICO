from app.vector_store import VectorStore
from app.db_manager import DBManager

class Dashboard:

    def __init__(self, vector_store: VectorStore, db_manager: DBManager):
        """
        Recibe el VectorStore y DBManager ya instanciado.
        Mismo patrón que RAGChain — no crea una nueva conexión.
        Estadisticias generales provinene de Supabase via db_manager
        Estadisticas de chunks siguen viniendo de ChromaDB via vector_store

        Input:  vector_store → instancia ya creada de VectorStore
        vector_store → instancia ya creada de DBManager
        Output: nada
        """
        self.vector_store = vector_store
        self.db_manager = db_manager

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

    def get_ml_stats(self) -> dict:
        """
        Estadísticas del modelo K-Means para el dashboard.

        Input:  nada
        Output: dict
                {
                  "docs_per_cluster": {
                    "Machine Learning":  3,
                    "Redes y Sistemas":  5,
                    "Sin clasificar":    2
                  },
                  "total_clusters": int
                }
        """
        try:
            docs = self.db_manager.get_all_document_stats()
            docs_per_cluster = {}
            for doc in docs:
                label = doc.get("cluster_label") or "Sin clasificar"
                if not label.strip():
                    label = "Sin clasificar"
                docs_per_cluster[label] = docs_per_cluster.get(label, 0) + 1
            total = sum(1 for k in docs_per_cluster if k != "Sin clasificar")
            return {"docs_per_cluster": docs_per_cluster, "total_clusters": total}
        except Exception as e:
            print(f"[Dashboard] Error obteniendo ML stats: {e}")
            return {"docs_per_cluster": {}, "total_clusters": 0}

    def get_usage_stats(self) -> dict:
        """
        Estadísticas de uso del sistema desde Supabase.
        Wrapper sobre db_manager.get_consultas_stats() con
        formato listo para st.metric() y st.bar_chart().

        Input:  nada
        Output: dict → output de db_manager.get_consultas_stats()
        """
        try:
            return self.db_manager.get_consultas_stats()
        except Exception as e:
            print(f"[Dashboard] Error obteniendo usage stats: {e}")
            return {
                "total_consultas":            0,
                "avg_response_time_ms":       0,
                "avg_similarity_score":       0,
                "consultas_sin_respuesta":    0,
                "consultas_por_dia":          {},
                "documentos_mas_consultados": []
            }