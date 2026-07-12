import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone, date
from config import DATABASE_URL

class DBManager:

    def __init__(self):
        """
        Establece conexión con Supabase usando DATABASE_URL del .env.
        Usa psycopg2 — misma librería que PostgreSQL local.

        Input:  nada
        Output: nada
        — self.conn:   conexión psycopg2 activa
        """
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            self.conn.autocommit = False
            print("[DBManager] Conexion a Supabase establecida")
        except psycopg2.OperationalError as e:
            raise ConnectionError(
                f"No se pudo conectarse con Supabase: {e}\n"
                f"Verifica DATABASE_URL en el archivo .env"
            )
    def _get_cursor(self):
        """
        Retorna cursor que devuelve dicts en vez de tuples.
        Función interna usada por todos los métodos.
        """
        return self.conn.cursor(cursor_factory=RealDictCursor)

    def _get_or_create_fecha(self, cursor, dt: datetime) -> int:
        """
        Busca o crea un registro en DIM_FECHA para la fecha dada.
        Función interna usada por insert_consulta().

        Input:  cursor → cursor activo de la transacción
                dt     → objeto datetime de la consulta
        Output: int → id_fecha
        """
        fecha = dt.date()
        cursor.execute(
            "SELECT id_fecha FROM dim_fecha WHERE fecha = %s",
            (fecha,)
        )
        row = cursor.fetchone()
        if row:
            return row["id_fecha"]
        #Mapeo de dia de la semana
        dias = {
            0: "Lunes", 1: "Martes", 2: "Miércoles",
            3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"
        }
        cursor.execute(
            """
            INSERT INTO dim_fecha (fecha, dia, mes, anio, dia_semana, trimestre)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id_fecha
            """,
            (
                fecha,
                fecha.day,
                fecha.month,
                fecha.year,
                dias[fecha.weekday()],
                (fecha.month - 1) // 3 + 1
            )
        )
        return cursor.fetchone()["id_fecha"]

    def _rows_to_dicts(self, cursor) ->list[dict]:
        """
         Convierte el resultado de fetchall() a lista de dicts.
         Con RealDictCursor esto es automático, pero esta función
         normaliza el tipo para garantizar list[dict] siempre.

         Input:  cursor → cursor después de un SELECT
         Output: list[dict]
         """
        rows = cursor.fetchall()
        return [dict(row) for row in rows]



    def initialize_schema(self) -> None:
        """
        Crea todas las tablas del modelo estrella si no existen.
        Usa CREATE TABLE IF NOT EXISTS — seguro llamarlo siempre.
        Llamada una sola vez en init_session_state().

        Tablas que crea:
          DIM_DOCUMENTOS         → un registro por PDF
          DIM_FECHA              → dimensión de tiempo
          DIM_RESULTADOS         → respuestas del LLM
          FACT_CONSULTAS         → tabla de hechos principal
          BRIDGE_CONSULTA_DOCS   → relación N:N consulta-documento

        Input:  nada
        Output: nada
        """
        schema_sql = """
            CREATE TABLE IF NOT EXISTS dim_documentos (
                id_documento    SERIAL PRIMARY KEY,
                source          VARCHAR(255) UNIQUE NOT NULL,
                title           VARCHAR(500),
                author          VARCHAR(255),
                subject         VARCHAR(255),
                keywords        TEXT,
                year            VARCHAR(10),
                metadata_source VARCHAR(20),
                total_chunks    INTEGER DEFAULT 0,
                total_pages     INTEGER DEFAULT 0,
                total_words     INTEGER DEFAULT 0,
                cluster_id      INTEGER,
                cluster_label   VARCHAR(100),
                upload_date     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS dim_fecha (
                id_fecha    SERIAL PRIMARY KEY,
                fecha       DATE NOT NULL UNIQUE,
                dia         INTEGER,
                mes         INTEGER,
                anio        INTEGER,
                dia_semana  VARCHAR(20),
                trimestre   INTEGER
            );

            CREATE TABLE IF NOT EXISTS dim_resultados (
                id_resultado      SERIAL PRIMARY KEY,
                respuesta         TEXT NOT NULL,
                chunks_retornados INTEGER,
                avg_similarity    FLOAT,
                sin_respuesta     BOOLEAN DEFAULT FALSE
            );

            CREATE TABLE IF NOT EXISTS fact_consultas (
                id_consulta      SERIAL PRIMARY KEY,
                pregunta         TEXT NOT NULL,
                id_resultado     INTEGER REFERENCES dim_resultados(id_resultado),
                id_fecha         INTEGER REFERENCES dim_fecha(id_fecha),
                response_time_ms INTEGER,
                timestamp        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS bridge_consulta_docs (
                id_consulta  INTEGER REFERENCES fact_consultas(id_consulta),
                id_documento INTEGER REFERENCES dim_documentos(id_documento),
                PRIMARY KEY (id_consulta, id_documento)
            );
        """
        try:
            with self.conn.get_cursor() as cursor:
                cursor.execute(schema_sql)
            self.conn.commit()
            print("[DBManager]Schema inicializado correctamente")
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Error al inicializar esquema: {e}")


    def upsert_document(self, metadata: dict,
                        stats: dict,
                        cluster_id: int = None,
                        cluster_label: str = None) -> int:
        """
        Inserta o actualiza documento en DIM_DOCUMENTOS.
        Usa INSERT ... ON CONFLICT DO UPDATE para re-indexaciones.

        Input:  metadata      → output de MetadataExtractor.extract()
                stats         → {"total_chunks": 47, "total_pages": 12,
                                  "total_words": 3420,
                                  "upload_date": "2024-01-15T10:32:00"}
                cluster_id    → int del cluster (None si aún no entrenado)
                cluster_label → string del cluster (None si aún no entrenado)
        Output: int → id_documento en la tabla
        """

    def insert_consulta(self, query: str,
                        answer: str,
                        sources: list[dict],
                        response_time_ms: int,
                        similarity_scores: list[float]) -> int:
        """
        Registra consulta RAG completa.
        Inserta en FACT_CONSULTAS, DIM_RESULTADOS y
        BRIDGE_CONSULTA_DOCS (una fila por documento fuente)
        Todos los parametros va dentro de una transacción  si falla algo, no queda registro parcial.

        Input:  query             → pregunta del usuario
                answer            → respuesta del LLM
                sources           → chunks usados como contexto
                                    [{"source": "tesis.pdf", "page": 3}, ...]
                response_time_ms  → tiempo de respuesta en ms
                similarity_scores → scores de similitud de cada chunk
        Output: int → id_consulta generado
        """

    def update_document_cluster(self, source: str,
                                cluster_id: int,
                                cluster_label: str) -> None:
        """
        Actualiza cluster_id Y cluster_label en DIM_DOCUMENTOS.
        Llamada para TODOS los documentos después de cada
        re-entrenamiento de K-Means.

        Input:  source        → nombre del archivo ("tesis.pdf")
                cluster_id    → número del cluster asignado
                cluster_label → label descriptivo ("Machine Learning")
        Output: nada
        """

    def sync_check(self) -> dict:
        """
        Compara documentos en ChromaDB vs Supabase y reporta
        inconsistencias sin resolverlas automáticamente.

        Input:  nada
        Output: dict con inconsistencias detectadas
                {
                  "in_chroma_not_supabase": ["nuevo.pdf"],
                  "in_supabase_not_chroma": ["borrado.pdf"],
                  "consistent":             ["tesis.pdf", "apuntes.pdf"]
                }
        """

    def get_all_document_stats(self) -> list[dict]:
        """
        Retorna estadísticas completas de todos los documentos.
        Usado por Dashboard y MLClassifier.

        Input:  nada
        Output: lista de dicts, uno por documento
                [
                  {
                    "id_documento":  1,
                    "source":        "tesis.pdf",
                    "title":         "Aprendizaje Automático",
                    "author":        "García, Juan",
                    "subject":       "Machine Learning",
                    "cluster_id":    0,
                    "cluster_label": "Machine Learning",
                    "total_chunks":  47,
                    "total_pages":   12,
                    "total_words":   3420,
                    "upload_date":   "2024-01-15"
                  }
                ]
        """

    def get_consultas_stats(self) -> dict:
        """
        Retorna estadísticas agregadas de consultas para el dashboard.

        Input:  nada
        Output: dict
                {
                  "total_consultas":            145,
                  "avg_response_time_ms":       1230,
                  "avg_similarity_score":       0.72,
                  "consultas_sin_respuesta":    8,
                  "consultas_por_dia":          {"2024-01-15": 12, ...},
                  "documentos_mas_consultados": [{"source": "tesis.pdf",
                                                  "count": 34}, ...]
                }
        """

    def execute_readonly_query(self, sql: str) -> list[dict]:
        """
        Ejecuta una query SQL generada por el LLM (Text-to-SQL).
        Solo permite SELECT — rechaza cualquier otra operación.
        Capa de seguridad adicional sobre los permisos de Supabase.

        Input:  sql → query generada por el LLM
        Output: list[dict] → filas retornadas como lista de dicts
        Raises: ValueError si la query no empieza con SELECT
        """

    def close(self) -> None:
        """
        Cierra la conexión con Supabase limpiamente.

        Input:  nada
        Output: nada
        """