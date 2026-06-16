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