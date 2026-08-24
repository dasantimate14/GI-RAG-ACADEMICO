from groq import Groq
import time
import numpy as np

from app.vector_store import VectorStore
from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K_SEMANTIC,
    TOP_K_KEYWORD,
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_MIN,
    TOP_K_FINAL
)


class RAGChain:

    def __init__(self, vector_store: VectorStore):
        """
        Inicializa el cliente de Groq y recibe el VectorStore.

        Input:  vector_store → instancia ya creada de VectorStore
                               (se recibe, no se crea aquí)
        Output: nada
        — self.client y self.vector_store quedan listos

        Nota: recibe VectorStore como parámetro (no lo instancia)
              porque main.py ya tiene una instancia creada.
              Esto evita crear dos conexiones a ChromaDB.
        """
        self.client = Groq(api_key=GROQ_API_KEY)
        self.vector_store = vector_store

    def _normalize_scores(self, chunks: list[dict], score_key: str = "distance") -> list[dict]:
        """
        Normaliza scores de un tipo de búsqueda a rango 0-1
        usando min-max normalization.

        Por qué es necesaria:
          BM25 retorna scores en rango variable (ej. 0-15.3)
          La búsqueda semántica retorna distances en 0-1
          Para fusionarlos en RRF necesitan estar en el mismo rango.

        Input:  chunks    → list[dict] con score en score_key
                score_key → str, key donde está el score (default "distance")
        Output: list[dict] → mismos chunks con score_key normalizado a 0-1
                             modifica los dicts in-place Y retorna la lista
        """
        if not chunks:
            return chunks

        scores = [chunk[score_key] for chunk in chunks]
        min_score = min(scores)
        max_score = max(scores)

        #Si todos los scores son iguales se asigna 1.0 a todos
        if max_score == min_score:
          for chunk in chunks:
              chunk[score_key] = 1.0
          return chunks

        #Min-Max normalization: (x-min) / (max-min)
        for chunk in chunks:
            chunk[score_key] = (chunk[score_key] - min_score) / (max_score - min_score)

        return chunks

    def _hybrid_search(self, query: str, filter_source: str = None) -> list[dict]:
        """
        Combina búsqueda semántica y BM25 usando
        Reciprocal Rank Fusion (RRF) para fusionar los rankings.

        Por qué RRF en vez de sumar scores directamente:
          Los scores de distintas fuentes no son comparables
          aunque los normalices. RRF usa las posiciones en
          el ranking (1°, 2°, 3°...) que son universalmente
          comparables entre cualquier método_ de búsqueda.
          Fórmula: RRF(d) = Σ 1/(k + rank(d)), k=60 estándar

        Input:  query         → pregunta del usuario
                filter_source → doc específico o None
        Output: list[dict] → chunks únicos ordenados por RRF score
                cada dict tiene "distance" = RRF score combinado
        """
        #Busqueda Semántica
        semantic_results = self.vector_store.search(
            query=query,
            filter_source=filter_source,
            n_results=TOP_K_SEMANTIC
        )

        #Busqueda BM25
        keywords_results = self.vector_store.search_by_keyword(
            query=query,
            filter_source=filter_source,
            n_results=TOP_K_KEYWORD
        )

        #Normaliza scores BM25
        keywords_results = self._normalize_scores(keywords_results)

        #RRF
        RRF_CONSTANT = 60  # constante estándar de RRF
        reciprocal_rank_fusion = {}  # key: chunk_id, value: {"chunk": dict, "score": float}

        for rank, chunk in enumerate(semantic_results):
            # Usa chunk_id de metadata como identificador único
            # Si no existe chunk_id, usa los primeros 80 chars del texto
            chunk_id = str(chunk["metadata"].get("chunk_id", chunk["text"][:80]))

            if chunk_id not in reciprocal_rank_fusion:
                reciprocal_rank_fusion[chunk_id] = {"chunk": chunk, "score": 0.0}
            reciprocal_rank_fusion[chunk_id]["score"] +=  1.0 / (RRF_CONSTANT + rank + 1)

        for rank, chunk in enumerate(keywords_results):
            chunk_id = str(chunk["metadata"].get("chunk_id", chunk["text"][:80]))

            if chunk_id not in reciprocal_rank_fusion:
                reciprocal_rank_fusion[chunk_id] = {"chunk": chunk, "score": 0.0}
            reciprocal_rank_fusion[chunk_id]["score"] += 1.0 / (RRF_CONSTANT + rank + 1)

        #Ordena por RRF score descendente
        merged = sorted(reciprocal_rank_fusion.values(), key=lambda x: x["score"], reverse=True)

        #Construye la lista final con RRF Score como "distance"

        results = []
        for item in merged:
            chunk = item["chunk"].copy()
            chunk["distance"] = item["score"]
            results.append(chunk)

        return results

    def _validate_and_rerank(self, chunks:list[dict], query:str) -> list[dict]:
        """
        Re-rankea chunks por similitud coseno real entre query y chunk,
        luego filtra los que no superan el umbral de calidad.

        Por qué re-rankear después de RRF:
          RRF combina rankings pero no mide similitud real.
          La similitud coseno entre el embedding de la query
          y el embedding del chunk es la métrica más directa
          de relevancia semántica real.

        Por qué similitud coseno y no distancia euclidiana:
          La similitud coseno mide el ángulo entre vectores,
          no la distancia. Para significado semántico el ángulo
          importa más que la magnitud. Es el estándar en NLP.

        Fórmula: similitud_coseno(a,b) = dot(a,b) / (||a|| * ||b||)
          Resultado en [-1, 1]:
            1.0  → vectores idénticos (mismo significado)
            0.0  → vectores ortogonales (sin relación)
           -1.0  → vectores opuestos (significado contrario)
          En embeddings de texto: prácticamente siempre entre 0 y 1

        Input:  chunks → list[dict] output de _hybrid_search()
                query  → pregunta original del usuario (str)
        Output: list[dict] → máximo TOP_K_FINAL chunks
                ordenados por cosine_similarity descendente
                todos con cosine_similarity >= SIMILARITY_THRESHOLD
                (o al menos 1 chunk si ninguno supera el umbral)
        """
        if not chunks:
            return []
        # Embedding del Query
        query_embedding = self.vector_store.embedder.generate_one(query)
        query_arr = np.array(query_embedding)

        #Calcula Similitud del Coseno para cada Chunk
        for chunk in chunks:
            chunk_embedding = self.vector_store.embedder.generate_one(
                chunk["text"]
            )
            chunk_arr = np.array(chunk_embedding)
            dot = np.dot(query_arr, chunk_arr)
            norm_query = np.linalg.norm(query_arr)
            norm_chunk = np.linalg.norm(chunk_arr)
            norm = norm_query * norm_chunk

            similarity = float(dot / norm) if norm > 0 else 0.0
            chunk["cosine_similarity"] = round(similarity, 4)

        #Ordena por similitud del coseno real descendente
        chunks_sorted = sorted(
            chunks,
            key=lambda x: x["cosine_similarity"],
            reverse=True
        )

        #Aplica umbral
        validated = [
            c for c in chunks_sorted
            if c["cosine_similarity"] >= SIMILARITY_THRESHOLD
        ]

        #Relaja el Umbral en caso de no obtener ningún resultado
        if len(validated) == 0:
            validated = [
                c for c in chunks_sorted
                if c["cosine_similarity"] >= SIMILARITY_THRESHOLD_MIN
            ]

        if len(validated) == 0 and chunks_sorted:
            validated = [chunks_sorted[0]]

        return validated[:TOP_K_FINAL]

    def build_prompt(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Construye el prompt que se enviará al LLM.
        Combina el contexto (chunks) con la pregunta del usuario.

        Input:  query  → pregunta del usuario
                chunks → output de VectorStore.search()
        Output: list[dict] → lista de mensajes con roles system y user listos para el LLM
        """
        context_block_list = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            source_name = meta.get("source", "Desconocido")
            pages = meta.get("pages", "N/A")
            page_start = meta.get("page_start", pages)
            relevancia = chunk.get("cosine_similarity", 0)

            context_block_list.append(
                f"Doc {i + 1} "
                f"(Archivo: {source_name}, "
                f"Páginas: {page_start}, "
                f"Relevancia: {relevancia:.2f}):"
                f"\n{chunk['text']}"
            )

        context_block = "\n\n".join(context_block_list)

        system_instruction = (
            "Eres un asistente de Q&A, basado en hechos. Tu objetivo es responder las preguntas del usuario "
            "usando solamente el contexto proporcionado. Sigue las siguientes reglas y restricciones:\n"
            "1. No uses conocimiento fuera del contexto.\n"
            "2. Si la respuesta no se puede encontrar en el contexto, responde exactamente con: "
            "'Disculpe. No pude encontrar la informacion solicitada en mi Base de Datos'.\n"
            "3. Cita tus fuentes concatenando el nombre del documento (e.g., [Doc 1]) a tus hechos.\n"
            "4. Si el contexto está en inglés y la pregunta en español, responde en español "
            "usando el contexto en inglés como fuente. Si un término técnico no tiene "
            "traducción común, mantenlo en inglés entre paréntesis junto a su traducción.\n"
            "5. Sé específico y detallado — no des respuestas vagas. "
            "Los documentos marcados con Relevancia más alta son más confiables; "
            "prioriza su contenido si hay contradicción entre fuentes."
        )

        user_instruction = (
            f"Context:\n"
            f"==========\n"
            f"{context_block}\n"
            f"=================\n\n"
            f"Query: {query}\n"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction}
        ]
        return messages



    def ask(self, query: str, filter_source: str = None) -> dict:
        """
        Función principal — orquesta el flujo RAG completo.
        Es la ÚNICA función que main.py necesita llamar de este módulo.

        Input:  query         → pregunta del usuario en texto plano
                filter_source → (opcional) limitar búsqueda a un PDF
        Output: dict con la respuesta y sus fuentes
                {
                  "answer":  "La regresión lineal es...",
                  "sources": [
                    {"source": "tesis.pdf", "page": 3},
                    {"source": "tesis.pdf", "page": 7}
                  ]
                }
        """
        inicio = time.time()

        #Hybrid Search (Semantica + BM25)
        candidates = self._hybrid_search(query, filter_source)

        #Re-ranking por similitud del coseno + validación
        top_chunks = self._validate_and_rerank(candidates, query)

        #Sin chunks válidos entonces responde sin llamar al LLM
        if not top_chunks:
            fin = time.time()
            return {
                "answer": "El contexto proporcionado no contiene "
                          "información sobre este tema.",
                "sources": [],
                "response_time_ms": int((fin - inicio) * 1000),
                "similarity_scores": []
            }

        #Construir prompt y llamar al LLM
        prompt = self.build_prompt(query=query, chunks=top_chunks)
        response = self.client.chat.completions.create(
            messages=prompt,
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=1000
        )
        answer = response.choices[0].message.content

        fin = time.time()

        #Extraer fuentes y scores para el retorno
        sources = [
            {
                "source": chunk["metadata"].get("source", ""),
                "page": chunk["metadata"].get("page_start", "?")
            }
            for chunk in top_chunks
        ]
        similarity_scores = [chunks["cosine_similarity"] for chunks in top_chunks]

        return {
            "answer": answer,
            "sources": sources,
            "response_time_ms": int((fin - inicio) * 1000),
            "similarity_scores": similarity_scores
        }
