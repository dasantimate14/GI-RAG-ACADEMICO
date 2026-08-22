from lib2to3.btm_utils import reduce_tree

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingsManager:

    def __init__(self):
        """
        Carga el modelo de sentence-transformers UNA SOLA VEZ.
        Usa EMBEDDING_MODEL de config.py.

        Input:  nada
        Output: nada
        — El modelo queda guardado en self.model
        """
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        Convierte una lista de textos en una lista de vectores.

        Input:  texts → lista de strings
                        ["chunk 1...", "chunk 2...", "pregunta del usuario..."]
        Output: lista de vectores numéricos
                [[0.12, -0.45, 0.88, ...], [0.33, 0.91, -0.22, ...], ...]
        """
        if not texts:
            raise ValueError("La lista de texto no puede estar vacia")
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def generate_one(self, text: str) -> list[float]:
        """
        Convierte un solo texto en un vector.
        Wrapper conveniente sobre generate() para queries individuales.

        Input:  text → string único (ej. pregunta del usuario)
        Output: vector numérico [0.12, -0.45, 0.88, ...]
        """
        return self.generate([text])[0]

    def get_dimension(self) -> int:
        """
        Retorna el número de dimensiones del modelo de embeddings activo.
        Necesaria para que vector_store pueda verificar compatibilidad
        con la colección existente en ChromaDB antes de insertar.

        Input:  nada
        Output: int → 768 para mpnet, 384 para MiniLM
        """
        return len(self.generate_one("dimension_check"))

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Procesa textos en lotes para mayor eficiencia.
        mpnet-base es más pesado que MiniLM — procesar
        en batches reduce el tiempo total de encoding.

        Input:  texts      → list[str] a encodear
                batch_size → int, default 32
        Output: list[list[float]] → un vector de 768 dims por texto
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


