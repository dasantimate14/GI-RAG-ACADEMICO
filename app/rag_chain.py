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

        Internamente:
          1. Llama a self.vector_store.search(query)
          2. Llama a self.build_prompt(query, chunks)
          3. Envía el prompt a Groq
          4. Retorna respuesta + fuentes
        """
        resultado_busqueda = self.vector_store.search(query, filter_source=filter_source)
        prompt = self.build_prompt(query=query, chunks=resultado_busqueda)

        chat_completion = self.client.chat.completions.create(
            messages=prompt,
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=1000
        )
        answer = chat_completion.choices[0].message.content

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
            "sources": sources
        }

