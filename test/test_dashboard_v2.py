import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.vector_store import VectorStore
from app.db_manager   import DBManager
from app.dashboard    import Dashboard

vs   = VectorStore()
db   = DBManager()

# Prueba 1 — __init__ acepta db_manager
dash = Dashboard(vs, db)
assert hasattr(dash, "vector_store"), "Falta self.vector_store"
assert hasattr(dash, "db_manager"),   "Falta self.db_manager"
print("OK: __init__ con db_manager OK")

# Prueba 2 — get_ml_stats() estructura correcta
ml_stats = dash.get_ml_stats()
assert "docs_per_cluster" in ml_stats, "Falta key 'docs_per_cluster'"
assert "total_clusters"   in ml_stats, "Falta key 'total_clusters'"
assert isinstance(ml_stats["docs_per_cluster"], dict)
assert isinstance(ml_stats["total_clusters"],   int)
print(f"OK: get_ml_stats OK -> {ml_stats}")

# Prueba 3 — documentos sin cluster van a "Sin clasificar"
try:
    docs_por_cluster    = ml_stats["docs_per_cluster"]
    total_docs_en_stats = sum(docs_por_cluster.values())
    total_docs_en_db    = len(db.get_all_document_stats())
    assert total_docs_en_stats == total_docs_en_db, \
           f"Todos los docs deben estar en algun cluster o 'Sin clasificar'. Stats={total_docs_en_stats}, DB={total_docs_en_db}"
    print("OK: Sin clasificar cubre docs sin cluster OK")
except RuntimeError as e:
    print(f"SKIP: Prueba 3 omitida (tabla Supabase no existe aun): {e}")

# Prueba 4 — get_usage_stats() estructura correcta
usage = dash.get_usage_stats()
expected_keys = [
    "total_consultas", "avg_response_time_ms",
    "avg_similarity_score", "consultas_sin_respuesta",
    "consultas_por_dia", "documentos_mas_consultados"
]
for key in expected_keys:
    assert key in usage, f"Falta key '{key}' en get_usage_stats()"
assert isinstance(usage["consultas_por_dia"],          dict)
assert isinstance(usage["documentos_mas_consultados"], list)
print(f"OK: get_usage_stats OK -> total consultas: {usage['total_consultas']}")

# Prueba 5 — get_usage_stats() no falla aunque Supabase este caido
class DBManagerRoto:
    def get_consultas_stats(self):
        raise ConnectionError("Supabase caido")
    def get_all_document_stats(self):
        return []

dash_roto  = Dashboard(vs, DBManagerRoto())
usage_safe = dash_roto.get_usage_stats()
assert usage_safe["total_consultas"] == 0
print("OK: get_usage_stats() es tolerante a fallos OK")
