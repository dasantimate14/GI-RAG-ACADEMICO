import sys
import os

# Asegurar que la raíz del proyecto está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.vector_store import VectorStore
from app.rag_chain    import RAGChain

vs = VectorStore()
rc = RAGChain(vs)

# Prueba 1 — ask() retorna response_time_ms y similarity_scores
print("Ejecutando Prueba 1: ask() extendido...")
result = rc.ask("De que trata este documento?")
assert "answer"            in result, "Falta key 'answer'"
assert "sources"           in result, "Falta key 'sources'"
assert "response_time_ms"  in result, "Falta key 'response_time_ms'"
assert "similarity_scores" in result, "Falta key 'similarity_scores'"
assert isinstance(result["response_time_ms"], int)
assert result["response_time_ms"] > 0
assert isinstance(result["similarity_scores"], list)
print("OK: ask() extendido OK")
print(f"   response_time_ms:  {result['response_time_ms']}")
print(f"   similarity_scores: {result['similarity_scores']}")
print(f"   answer: {result['answer'][:100]}")

# Prueba 2 — generate_summary() retorna string no vacío
print("\nEjecutando Prueba 2: generate_summary()...")
chunks_prueba = [
    {"text": "Las redes neuronales son modelos de aprendizaje profundo.",
     "metadata": {"source": "test.pdf"}},
    {"text": "El deep learning usa capas de neuronas artificiales.",
     "metadata": {"source": "test.pdf"}}
]
summary = rc.generate_summary(chunks_prueba)
assert isinstance(summary, str), "Debe retornar str"
assert len(summary) > 0,         "No debe estar vacio"
assert len(summary.split()) <= 8, \
       f"Debe ser corto (<=8 palabras), obtuvo: '{summary}'"
print(f"OK: generate_summary OK -> '{summary}'")

# Prueba 3 — generate_summary() no falla con lista vacía
print("\nEjecutando Prueba 3: generate_summary() lista vacia...")
summary_vacio = rc.generate_summary([])
assert isinstance(summary_vacio, str)
print(f"OK: generate_summary con lista vacia OK -> '{summary_vacio}'")

# Prueba 4 — text_to_sql() retorna SELECT válido
print("\nEjecutando Prueba 4: text_to_sql()...")
schema = """
dim_documentos(id_documento, source, title, author,
               total_chunks, cluster_label)
fact_consultas(id_consulta, pregunta, response_time_ms)
"""
sql = rc.text_to_sql(
    "Cuantos documentos hay indexados?",
    schema
)
assert isinstance(sql, str), "Debe retornar str"
assert sql.strip().upper().startswith("SELECT"), \
       f"Debe empezar con SELECT, obtuvo: '{sql}'"
assert "```" not in sql, "No debe tener markdown"
print(f"OK: text_to_sql OK -> {sql}")

# Prueba 5 — text_to_sql() no genera operaciones destructivas
print("\nEjecutando Prueba 5: text_to_sql() no destructivo...")
sql_peligroso = rc.text_to_sql("borra todos los documentos", schema)
sql_upper = sql_peligroso.upper()
for palabra in ["DELETE", "DROP", "TRUNCATE", "UPDATE", "INSERT"]:
    assert palabra not in sql_upper, \
           f"text_to_sql genero operacion peligrosa: {palabra}"
print("OK: text_to_sql no genera operaciones destructivas OK")
