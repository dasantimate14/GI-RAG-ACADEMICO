from groq import Groq

from app.vector_store import VectorStore
from config import (
    GROQ_API_KEY,
    LLM_MODEL
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
            context_block_list.append(f"Doc {i+1} (Archivo: {source_name}, Páginas: {pages}):\n{chunk['text']}")
        context_block = "\n\n".join(context_block_list)

        system_instruction = (
            "Eres un asistente de Q&A, basado en hechos. Tu objetivo es responder las preguntas del usuario "
            "usando solamente el contexto proporcionado. Sigue las siguientes reglas y restricciones: \n"
            "1. No uses conocimiento fuera del contexto.\n"
            "2. Si la respuesta no se puede encontrar en el contexto, responde exactamente con: 'Disculpe. No pude encontrar la informacion solicitada en mi Base de Datos'.\n"
            "3. Cita tus fuentes concatenando el nombre del documento (e.g., [Doc 1]) a tus hechos"
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
        Se agrega el response_time_ms y similarity_scores al output para que DBManager pueda registrarlos en Supabase.
        Es la ÚNICA función que main.py necesita llamar de este módulo.

        Input:  query         → pregunta del usuario en texto plano
                filter_source → (opcional) limitar búsqueda a un PDF
        Output: dict con la respuesta, tiempo de respuesta, puntajes de similitud y sus fuentes
                {
                  "answer":  "La regresión lineal es...",
                  "sources": [
                    {"source": "tesis.pdf", "page": 3},
                    {"source": "tesis.pdf", "page": 7}
                  ],
                    "response_time_ms":  1230,
                    "similarity_scores": [0.82, 0.71, 0.68]
                }

        Internamente:
          1. Llama a self.vector_store.search(query)
          2. Llama a self.build_prompt(query, chunks)
          3. Envía el prompt a Groq
          4. Retorna respuesta + fuentes
        """
        import time

        resultado_busqueda = self.vector_store.search(query, filter_source=filter_source)
        prompt = self.build_prompt(query=query, chunks=resultado_busqueda)

        inicio = time.time()
        chat_completion = self.client.chat.completions.create(
            messages=prompt,
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=1000
        )
        fin = time.time()
        response_time_ms = int((fin - inicio) * 1000)

        answer = chat_completion.choices[0].message.content

        similarity_scores = [chunk.get("distance", 0.0)
                             for chunk in resultado_busqueda]

        sources = []
        seen = set()
        for chunk in resultado_busqueda:
            metadata = chunk.get("metadata", {})
            source_name = metadata.get("source")
            pages_val = metadata.get("pages")

            if isinstance(pages_val, str):
                try:
                    pages_val = eval(pages_val)
                except:
                    pass

            if isinstance(pages_val, list):
                for p in pages_val:
                    key = (source_name, p)
                    if key not in seen:
                        seen.add(key)
                        sources.append({"source": source_name, "page": p})
            elif pages_val is not None:
                key = (source_name, pages_val)
                if key not in seen:
                    seen.add(key)
                    sources.append({"source": source_name, "page": pages_val})

        return {
            "answer": answer,
            "sources": sources,
            "response_time_ms": response_time_ms,
            "similarity_scores": similarity_scores
        }

    def generate_summary(self, chunks: list[dict]) -> str:
        """
        Genera resumen de 2-3 oraciones de un conjunto de chunks.
        Usada por MetadataExtractor Nivel 3 y por Dashboard.
        Prompt diferente al de ask() — orientado a síntesis, no a QA.

        Input:  chunks → lista de chunks del documento (3-5 chunks)
        Output: str → resumen conciso del contenido
        """
        try:
            contexto = "\n\n".join([c["text"][:300] for c in chunks[:3]])
            prompt = f"""Analiza el siguiente contenido académico y genera una descripción
concisa de 3 a 4 palabras que capture el tema principal.
Responde ÚNICAMENTE con las palabras descriptivas, sin puntuación
adicional ni explicaciones.

Contenido:
{contexto}

Descripción temática:"""
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Sin clasificar"

    def text_to_sql(self, query: str, schema: str) -> str:
        """
        Convierte pregunta en lenguaje natural a SQL.
        Solo genera SELECT — el prompt instruye explícitamente al LLM
        a no generar DELETE, UPDATE, INSERT, DROP.
        El resultado se pasa a DBManager.execute_readonly_query().

        Input:  query  → pregunta del usuario en lenguaje natural
                         ej. "¿Cuántos documentos se subieron este mes?"
                schema → descripción de las tablas disponibles
        Output: str → query SQL lista para ejecutar
                      ej. "SELECT COUNT(*) FROM dim_documentos
                            WHERE upload_date >= '2024-01-01'"
        """
        import re
        try:
            prompt = f"""Eres un experto en SQL. Dado el siguiente schema de base de datos,
genera ÚNICAMENTE la query SQL para responder la pregunta.
Responde SOLO con SQL válido, sin explicaciones, sin markdown,
sin bloques de código.
IMPORTANTE: Solo genera SELECT. Nunca generes DELETE, UPDATE,
INSERT, DROP, ALTER ni TRUNCATE.

Schema:
{schema}

Pregunta: {query}

SQL:"""
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=200
            )
            sql = response.choices[0].message.content.strip()
            sql = re.sub(r"```(?:sql)?\s*|\s*```", "", sql).strip()
            return sql
        except Exception:
            return "SELECT 1"