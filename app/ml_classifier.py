import numpy as np
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from app.vector_store import VectorStore
from app.rag_chain import RAGChain

from config import (
    ML_MODEL_PATH,
    ML_MIN_DOCS_FOR_TRAINING,
    ML_MAX_CLUSTERS
)

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
        self.vector_store    = vector_store
        self.rag_chain       = rag_chain
        self.model           = None
        self.is_trained      = False
        self.cluster_labels  = {}   # {cluster_id (int): label (str)}
        self.train_result    = {}   # guarda el último resultado de train()

        os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
        self.load()

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
        if not chunk_embeddings:
            raise ValueError(
                "chunk_embeddings no puede estar vacío"
            )
        embbedings_array = np.array(chunk_embeddings)
        mean_embeddings = np.mean(embbedings_array, axis=0)
        return mean_embeddings.tolist()

    def _get_optimal_k(self,
                       embeddings: list[list[float]]) -> int:
        """
        Función interna — determina el k óptimo usando
        silhouette score entre k=2 y k=min(8, n_docs-1).
        Llamada dentro de train() antes de entrenar.

        Input:  embeddings → lista de vectores de documentos
        Output: int → k óptimo recomendado
        """
        n_docs = len(embeddings)
        k_max = min(ML_MAX_CLUSTERS, n_docs//2)

        if k_max < 2:
            return 2

        embeddings_array = np.array(embeddings)
        best_k = 2
        best_score = -1.0

        # Se evalua para cada k en el rango permitido (k_max + 1 para que sea inclusivo)
        for k in range(2, k_max + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                max_iter=300,
            )
            labels = kmeans.fit_predict(embeddings_array)

            #Calcula la metrica de cohesion y separacion de los clusters
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue

            score = silhouette_score(embeddings_array, labels)

            #Si el score es mejor entonces actualizamos el k optimo
            if score > best_score:
                best_score = score
                best_k = k

        return best_k


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
        #Filtrar los documentos que pertenecen al cluster especifico
        cluster_docs = [
            source for source, data in assignments.items()
            if data["cluster_id"] == cluster_id
        ]
        if not cluster_docs:
            return f"Cluster {cluster_id}"

        representative_chunks = []

        #Se limita a solo los primeros tres documentos para no saturar el contexto
        for source in cluster_docs[:3]:
            try:
                #Toma los primeros 2 chunks de texto de este documento
                results = self.vector_store.search(
                    query=source,
                    filter_source=source
                )
                for r in results[:2]:
                    representative_chunks.append({
                        "text": r["text"],
                        "metadata": r["metadata"]
                    })
            except Exception:
                #Si un documento falla continua con los demas
                continue
        if not representative_chunks:
            return f"Cluster {cluster_id}"

        try:
            label = self.rag_chain.generate_summary(representative_chunks)
            label = label.strip().rstrip(".,;:")
            words = label.split()
            if len(words) > 5:
                label = " ".join(words[:5])
            return label if label else f"Cluster {cluster_id}"
        except Exception:
            return f"Cluster {cluster_id}"


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
        n_docs = len(documents_embeddings)

        #Se verifica si existe el minimo de documentos necesarios
        if n_docs < ML_MIN_DOCS_FOR_TRAINING:
            return {
                "status": "insufficient_data",
                "n_docs": n_docs,
                "min_required": ML_MIN_DOCS_FOR_TRAINING,
                "message": f"Se necesitan al menos {ML_MIN_DOCS_FOR_TRAINING} documentos" 
                            f"Actualmente solo hay {n_docs}."
            }
        sources = list(documents_embeddings.keys())
        embeddings = list(documents_embeddings.values())
        embeddings_array = np.array(embeddings)

        #Se determina la K optima
        optimal_k = self._get_optimal_k(embeddings)

        #Se entrena el K-Means
        self.model = KMeans(
            n_clusters=optimal_k,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.model.fit(embeddings_array)
        self.is_trained = True

        labels = self.model.labels_
        sil_score = silhouette_score(embeddings_array, labels)
        sil_samples = silhouette_samples(embeddings_array, labels)

        #Se construyen asignaciones preliminares sin etiquetas descriptivas
        assignments = {}
        for i, source in enumerate(sources):
            assignments[source] = {
                "cluster_id": int(labels[i]),
                "cluster_label": "",
                "silhouette_sample": float(sil_samples[i])
            }

        #Genera labels para cada cluster
        unique_clusters_ids = set(int(l) for l in labels)
        self.cluster_labels = {}

        for cluster_id in unique_clusters_ids:
            label = self._generate_cluster_label(cluster_id, assignments)
            self.cluster_labels[cluster_id] = label

        #Se agregan los labels a las asignaciones
        for source in assignments:
            cluster_id = assignments[source]["cluster_id"]
            assignments[source]["cluster_label"] = self.cluster_labels[cluster_id]

        #Se construye la salida final
        docs_per_cluster = {}
        for cluster_id, label in self.cluster_labels.items():
            count = sum(
                1 for info in assignments.values()
                if info["cluster_id"] == cluster_id
            )
            docs_per_cluster[label] = count

        self.train_result = {
            "status": "trained",
            "n_clusters": optimal_k,
            "silhouette_score": round(float(sil_score), 4),
            "docs_per_cluster": docs_per_cluster,
            "cluster_assignments": assignments
        }
        self.save()
        return self.train_result

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
        if not self.is_trained or self.model is None:
            raise RuntimeError(
                "El modelo no esta entrenado.\nLLama a train() "
                "o verifica que el modelo se cargo correctamente desde el disco"
            )
        embedding_array = np.array([document_embedding])
        cluster_id = int(self.model.predict(embedding_array)[0])
        label = self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        return {
            "cluster_id": cluster_id,
            "cluster_label": label
        }


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
        sources = self.vector_store.get_all_sources()
        if not sources:
            return {}

        documents_embeddings = {}

        for source in sources:
            try:
                chunk_embeddings = self.vector_store.get_document_embedding(source)

                if not chunk_embeddings:
                    continue

                doc_embedding = self._compute_document_embedding(chunk_embeddings)
                documents_embeddings[source] = doc_embedding
            except Exception as e:
                print(f"[MLClassifier] Error procesando {source}: {e}")
                continue

        return documents_embeddings

    def save(self) -> None:
        """
        Persiste el modelo K-Means en disco como archivo .pkl.
        Usa ML_MODEL_PATH de config.py.
        Llamada automáticamente al final de train().

        Input:  nada
        Output: nada — escribe archivo en disco
        """
        if self.model is None:
            return

        payload = {
            "model": self.model,
            "cluster_labels": self.cluster_labels,
            "train_result": self.train_result
        }
        joblib.dump(payload, ML_MODEL_PATH)
        print(f"[MLClassifier] Modelo guardado en {ML_MODEL_PATH}")

    def load(self) -> bool:
        """
        Carga modelo K-Means desde disco si existe.
        Llamada en __init__() automáticamente.

        Input:  nada
        Output: bool → True si se cargó, False si no existía
        """
        if not os.path.exists(ML_MODEL_PATH):
            return False
        try:
            payload = joblib.load(ML_MODEL_PATH)
            self.model = payload["model"]
            self.cluster_labels = payload["cluster_labels"]
            self.train_result = payload["train_result"]
            self.is_trained = True
            print(f"[MLClassifier] Modelo cargado desde {ML_MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[MLClassifier] Error cargando modelo: {e}")
            self.model = None
            self.is_trained = False
            return False