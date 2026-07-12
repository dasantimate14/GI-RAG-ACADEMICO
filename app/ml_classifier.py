from app.vector_store import VectorStore
from app.rag_chain import RAGChain
class MLClassifier:

    def __init__(self, vector_store: VectorStore,
                 rag_chain: RAGChain):
        """
        Recibe VectorStore y RAGChain ya instanciados.
        Intenta cargar modelo K-Means desde disco si existe.

        Input:  vector_store → para obtener embeddings de chunks
                rag_chain    → para generar cluster labels con LLM
        Output: nada
        — self.model:      KMeans o None
        — self.is_trained: bool
        — self.vector_store y self.rag_chain guardados
        """
        self.vector_store = vector_store
        self.rag_chain = rag_chain

    def _compute_document_embedding(self,
                                    chunk_embeddings: list[list[float]]
                                    ) -> list[float]:
        """
        Función interna — promedia todos los chunk embeddings
        de un documento para obtener un vector representativo único.
        No la llama nadie fuera de esta clase.

        Input:  chunk_embeddings → lista de vectores de los chunks
        Output: list[float] → vector promedio del documento
        """

    def _get_optimal_k(self,
                       embeddings: list[list[float]]) -> int:
        """
        Función interna — determina el k óptimo usando
        silhouette score entre k=2 y k=min(8, n_docs-1).
        Llamada dentro de train() antes de entrenar.

        Input:  embeddings → lista de vectores de documentos
        Output: int → k óptimo recomendado
        """

    def _generate_cluster_label(self,
                                 cluster_id: int,
                                 assignments: dict) -> str:
        """
        Función interna — genera label descriptivo para un cluster
        pasando chunks representativos al LLM via rag_chain.generate_summary().
        Llamada dentro de train() para cada cluster formado.

        Input:  cluster_id  → número del cluster (0, 1, 2...)
                assignments → dict completo de asignaciones
                              {"tesis.pdf": {"cluster_id": 0, ...}, ...}
        Output: str → label descriptivo de 3-4 palabras
                      ej. "Machine Learning", "Redes y Sistemas"
        """

    def train(self, documents_embeddings: dict) -> dict:
        """
        Entrena K-Means con embeddings de TODOS los documentos.
        SOLO ejecuta si hay 4 o más documentos — si no, retorna
        status "insufficient_data" y no modifica nada.
        Llama internamente a _get_optimal_k() y _generate_cluster_label().
        Persiste el modelo con save() al finalizar.

        Input:  documents_embeddings → dict key=source, value=embedding promedio
                {
                  "tesis.pdf":   [0.12, -0.45, ...],
                  "apuntes.pdf": [0.33, 0.91, ...]
                }
        Output: dict con resultados
                {
                  "status":          "trained" | "insufficient_data",
                  "n_clusters":      3,
                  "silhouette_score": 0.68,
                  "cluster_assignments": {
                    "tesis.pdf": {
                      "cluster_id":    0,
                      "cluster_label": "Machine Learning"
                    },
                    "apuntes.pdf": {
                      "cluster_id":    1,
                      "cluster_label": "Redes y Sistemas"
                    }
                  }
                }
        """

    def predict(self, document_embedding: list[float]) -> dict:
        """
        Predice cluster de un documento nuevo con modelo ya entrenado.

        Input:  document_embedding → vector promedio del documento
        Output: dict
                {
                  "cluster_id":    0,
                  "cluster_label": "Machine Learning"
                }
        Raises: RuntimeError si is_trained es False
        """

    def get_all_document_embeddings(self) -> dict:
        """
        Obtiene embeddings promedio de TODOS los documentos
        actualmente en ChromaDB para pasarlos a train().
        Orquesta llamadas a vector_store.get_all_sources()
        y vector_store.get_document_embeddings() por cada fuente.

        Input:  nada
        Output: dict key=source, value=embedding promedio
                {
                  "tesis.pdf":   [0.12, -0.45, ...],
                  "apuntes.pdf": [0.33, 0.91, ...]
                }
        """

    def save(self) -> None:
        """
        Persiste el modelo K-Means en disco como archivo .pkl.
        Usa ML_MODEL_PATH de config.py.
        Llamada automáticamente al final de train().

        Input:  nada
        Output: nada — escribe archivo en disco
        """

    def load(self) -> bool:
        """
        Carga modelo K-Means desde disco si existe.
        Llamada en __init__() automáticamente.

        Input:  nada
        Output: bool → True si se cargó, False si no existía
        """