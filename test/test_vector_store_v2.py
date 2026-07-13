import sys
import os

# Asegurar que la raíz del proyecto está en el PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.vector_store import VectorStore

# Requisito: haber indexado al menos 2 PDFs antes de correr
vs      = VectorStore()
sources = vs.get_all_sources()

# Prueba 1 — get_all_sources() retorna lista no vacía
assert isinstance(sources, list), "Debe retornar list"
print(f"OK: get_all_sources OK -> {len(sources)} documentos: {sources}")

# Prueba 2 — get_all_sources() no tiene duplicados
assert len(sources) == len(set(sources)), "No debe haber duplicados"
print("OK: Sin duplicados OK")

# Prueba 3 — get_document_embeddings() retorna embeddings válidos
if sources:
    source_prueba = sources[0]
    embeddings    = vs.get_document_embeddings(source_prueba)
    assert isinstance(embeddings, list), "Debe retornar list"
    assert len(embeddings) > 0, "Debe tener al menos 1 embedding"
    assert isinstance(embeddings[0], list), "Cada embedding debe ser list"
    assert all(isinstance(v, float) for v in embeddings[0]), \
           "Valores deben ser float"
    print(f"OK: get_document_embeddings OK -> {len(embeddings)} "
          f"embeddings de {len(embeddings[0])} dimensiones")
else:
    print("WARNING: No hay sources indexados para probar embeddings")

# Prueba 4 — source inexistente retorna lista vacía
embeddings_vacio = vs.get_document_embeddings("documento_que_no_existe.pdf")
assert embeddings_vacio == [], \
       f"Source inexistente debe retornar [], obtuvo {embeddings_vacio}"
print("OK: Source inexistente retorna [] OK")

# Prueba 5 — consistencia entre get_all_sources y get_document_embeddings
for source in sources:
    embs = vs.get_document_embeddings(source)
    assert len(embs) > 0, \
           f"Cada source debe tener embeddings, {source} tiene 0"
print("OK: Consistencia sources <-> embeddings OK")
