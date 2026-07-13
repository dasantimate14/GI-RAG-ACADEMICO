# Ejecutar con: python test/test_db_manager.py
import sys
sys.path.append(".")
from app.db_manager import DBManager

def test_connection():
    db = DBManager()
    assert db.conn is not None
    assert not db.conn.closed
    print("✅ Conexión OK")
    db.close()

def test_initialize_schema():
    db = DBManager()
    db.initialize_schema()    # no debe lanzar excepción
    db.initialize_schema()    # segunda llamada también debe ser segura
    print("✅ initialize_schema OK (idempotente)")
    db.close()

def test_upsert_document():
    db = DBManager()
    db.initialize_schema()

    metadata = {
        "source":          "test_paper.pdf",
        "title":           "Prueba de Base de Datos",
        "author":          "García, Juan",
        "subject":         "Bases de Datos",
        "keywords":        "SQL, PostgreSQL",
        "year":            "2024",
        "metadata_source": "pdf"
    }
    stats = {
        "total_chunks": 10,
        "total_pages":  5,
        "total_words":  1200,
        "upload_date":  "2024-01-15T10:00:00+00:00"
    }

    id1 = db.upsert_document(metadata, stats)
    assert isinstance(id1, int) and id1 > 0
    print(f"✅ upsert_document (insert) OK — id: {id1}")

    # Segunda llamada debe actualizar, no duplicar
    metadata["title"] = "Título Actualizado"
    id2 = db.upsert_document(metadata, stats, cluster_id=0,
                              cluster_label="Bases de Datos")
    assert id1 == id2    # mismo id → fue UPDATE no INSERT
    print(f"✅ upsert_document (update) OK — mismo id: {id2}")
    db.close()

def test_insert_consulta():
    db = DBManager()
    db.initialize_schema()

    id_consulta = db.insert_consulta(
        query="¿Qué es SQL?",
        answer="SQL es un lenguaje de consulta estructurado.",
        sources=[{"source": "test_paper.pdf", "page": 1}],
        response_time_ms=850,
        similarity_scores=[0.82, 0.74]
    )
    assert isinstance(id_consulta, int) and id_consulta > 0
    print(f"✅ insert_consulta OK — id: {id_consulta}")
    db.close()

def test_execute_readonly_query():
    db = DBManager()
    db.initialize_schema()

    # Query válida
    results = db.execute_readonly_query(
        "SELECT COUNT(*) AS total FROM dim_documentos"
    )
    assert isinstance(results, list)
    print(f"✅ execute_readonly_query OK — resultado: {results}")

    # Query inválida debe lanzar ValueError
    try:
        db.execute_readonly_query("DELETE FROM dim_documentos")
        assert False, "Debió lanzar ValueError"
    except ValueError:
        print("✅ execute_readonly_query rechaza DELETE OK")

    db.close()

def test_sync_check():
    db = DBManager()
    db.initialize_schema()

    resultado = db.sync_check(["test_paper.pdf", "doc_nuevo.pdf"])
    assert "in_chroma_not_supabase" in resultado
    assert "in_supabase_not_chroma" in resultado
    assert "consistent" in resultado
    assert "doc_nuevo.pdf" in resultado["in_chroma_not_supabase"]
    print(f"✅ sync_check OK — {resultado}")
    db.close()

def test_get_consultas_stats():
    db = DBManager()
    db.initialize_schema()

    stats = db.get_consultas_stats()
    assert "total_consultas" in stats
    assert "avg_response_time_ms" in stats
    assert "consultas_por_dia" in stats
    assert "documentos_mas_consultados" in stats
    print(f"✅ get_consultas_stats OK — {stats}")
    db.close()

if __name__ == "__main__":
    test_connection()
    test_initialize_schema()
    test_upsert_document()
    test_insert_consulta()
    test_execute_readonly_query()
    test_sync_check()
    test_get_consultas_stats()
    print("\n✅ Todos los tests pasaron")