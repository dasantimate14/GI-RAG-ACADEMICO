import streamlit as st
import os
from app.db_manager import DBManager
from app.pdf_processor import PDFProcessor
from app.vector_store import VectorStore
from app.rag_chain import RAGChain
from app.ml_classifier import MLClassifier
from app.metada_extractor import MetadataExtractor
from app.dashboard import Dashboard
from config import UPLOAD_PATH


def init_session_state():
    """
    Inicializa todas las variables de st.session_state si no existen.
    Debe llamarse al inicio de main.py, antes de cualquier otra cosa.

    Variables que inicializa:
      - messages: []              → historial del chat
      - vector_store: VectorStore → instancia compartida entre tabs
      - rag_chain: RAGChain       → instancia del RAG
      - dashboard: Dashboard      → instancia del dashboard
      - db_manager: DBManager     → instancia del DBManager
      - ml_classifier: ML_Classifier → instancia del ML_Classifier
      -metadata_extractor: Metadata_Extractor → instancia del Metadata_Extractor
      -pdf_processor: PDFProcessor →  instancia del PDFProcessor

    Orden de inicialización estricto por dependencias:
    1. db_manager         → DBManager() db_manager.initialize_schema()
    2. vector_store       → VectorStore()
    3. rag_chain          → RAGChain(vector_store)
    4. ml_classifier      → MLClassifier(vector_store, rag_chain)
    5. metadata_extractor → MetadataExtractor(rag_chain)
    6. pdf_processor      → PDFProcessor()
    7. dashboard          → Dashboard(vector_store, db_manager)

    Nota: al usar st.session_state, las instancias se crean UNA
          sola vez aunque Streamlit re-ejecute el script completo.
    """
    if "initialized" not in st.session_state:

        # 1. DBManager — sin dependencias
        st.session_state.db_manager = DBManager()
        st.session_state.db_manager.initialize_schema()

        # 2. VectorStore — sin dependencias
        st.session_state.vector_store = VectorStore()

        # 3. RAGChain — depende de vector_store
        st.session_state.rag_chain = RAGChain(
            st.session_state.vector_store
        )

        # 4. MLClassifier — depende de vector_store y rag_chain
        st.session_state.ml_classifier = MLClassifier(
            st.session_state.vector_store,
            st.session_state.rag_chain
        )

        # 5. MetadataExtractor — depende de rag_chain
        st.session_state.metadata_extractor = MetadataExtractor(
            st.session_state.rag_chain
        )

        # 6. PDFProcessor — recibe metadata_extractor
        st.session_state.pdf_processor = PDFProcessor()

        # 7. Dashboard — depende de vector_store y db_manager
        st.session_state.dashboard = Dashboard(
            st.session_state.vector_store,
            st.session_state.db_manager
        )

        # 8. Historial del chat
        st.session_state.messages    = []
        st.session_state.initialized = True


def render_sidebar():
    """
    Renderiza el sidebar con el uploader de PDFs y lista de documentos.
    Pipeline completo al subir PDF:

    1. pdf_processor.process(file, metadata_extractor)
       → retorna {chunks, metadata, stats}
    2. vector_store.document_exists(source)
       → si existe, pregunta al usuario si re-indexar
    3. vector_store.add_documents(chunks)
    4. try: db_manager.upsert_document(metadata, stats)
       except: log error, continuar
    5. ml_classifier.get_all_document_embeddings()
    6. ml_classifier.train(embeddings)
       → si status="insufficient_data", skip pasos 7-8
    7. for each doc in assignments:
           try: db_manager.update_document_cluster(source, id, label)
           except: log error, continuar
    8. st.success("Documento indexado correctamente")

    Input:  nada (lee de st.session_state)
    Output: nada (escribe en st.session_state si se sube un PDF)

    Contiene:
      - st.file_uploader() para subir PDFs
      - Lógica de indexación al subir un archivo
      - Lista de documentos ya indexados
    """
    with st.sidebar:
        st.title("📂 Gestión de Documentos")

        uploaded_files = st.file_uploader(
            "Cargar archivos PDF para indexar",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner("Procesando documento..."):
                    try:
                        # Paso 1: verifica si ya existe
                        source = uploaded_file.name
                        if st.session_state.vector_store.document_exists(source):
                            st.warning(f"'{source}' ya está indexado. "
                                       f"Se re-indexará.")
                            st.session_state.vector_store.delete_document(source)

                        # Paso 2: procesa el PDF (chunks + metadata + stats)
                        result   = st.session_state.pdf_processor.process(
                            uploaded_file,
                            st.session_state.metadata_extractor
                        )
                        chunks   = result["chunks"]
                        metadata = result["metadata"]
                        stats    = result["stats"]

                        if not chunks:
                            st.error("No se pudo extraer texto del PDF. "
                                     "¿Es un PDF escaneado?")
                            st.stop()

                        # Paso 3: indexa en ChromaDB
                        n_indexed = st.session_state.vector_store.add_documents(chunks)

                        # Paso 4: registra en Supabase (tolerante a fallos)
                        try:
                            st.session_state.db_manager.upsert_document(
                                metadata=metadata,
                                stats=stats
                            )
                        except Exception as e:
                            st.warning(f"Documento indexado pero no registrado "
                                       f"en base de datos: {e}")

                        # Paso 5: re-entrena K-Means
                        all_embeddings = (st.session_state.ml_classifier
                                            .get_all_document_embeddings())
                        train_result   = st.session_state.ml_classifier.train(
                                             all_embeddings
                                         )

                        # Paso 6: actualiza clusters en Supabase si entrenó
                        if train_result.get("status") == "trained":
                            assignments = train_result.get("cluster_assignments", {})
                            for doc_source, info in assignments.items():
                                try:
                                    st.session_state.db_manager.update_document_cluster(
                                        source        = doc_source,
                                        cluster_id    = info["cluster_id"],
                                        cluster_label = info["cluster_label"]
                                    )
                                except Exception as e:
                                    print(f"[main] Error actualizando cluster "
                                          f"de {doc_source}: {e}")
                        elif train_result.get("status") == "insufficient_data":
                            st.info(f"Se necesitan al menos "
                                    f"{train_result['min_required']} documentos "
                                    f"para clasificar automáticamente. "
                                    f"Actualmente hay {train_result['n_docs']}.")

                        st.success(
                            f"✅ '{source}' indexado correctamente. "
                            f"{stats['total_chunks']} chunks · "
                            f"{stats['total_pages']} páginas · "
                            f"{stats['total_words']} palabras"
                        )

                    except Exception as e:
                        st.error(f"Error procesando '{uploaded_file.name}': {e}")

        st.markdown("---")
        st.subheader("📚 Documentos Indexados")

        docs_metadata = st.session_state.vector_store.get_all_documents()
        if not docs_metadata:
            st.info("No hay documentos indexados aún.")
        else:
            for doc in docs_metadata:
                source = doc["source"]
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(f"📄 **{source}** ({doc['total_chunks']} chunks, {doc['pages']} págs)")
                with col2:
                    if st.button("🗑️", key=f"del_{source}"):
                        st.session_state.vector_store.delete_document(source)
                        file_path = os.path.join(UPLOAD_PATH, source)
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except:
                                pass
                        st.success(f"Eliminado: {source}")
                        st.rerun()


def render_chat_tab():
    """
    Renderiza la pestaña del chatbot.
    Registra consulta en Supabase después de responder:

    1. result = rag_chain.ask(query)
    2. Muestra result["answer"] al usuario inmediatamente
    3. try:
           db_manager.insert_consulta(
               query, answer, sources,
               response_time_ms, similarity_scores
           )
       except: log error   ← usuario nunca ve este error

    Input:  nada (lee de st.session_state)
    Output: nada (actualiza st.session_state.messages)

    Contiene:
      - Historial de mensajes con st.chat_message()
      - Input del usuario con st.chat_input()
      - Llamada a rag_chain.ask() y muestra de fuentes
    """
    st.header("💬 Chatbot Académico")

    docs_metadata = st.session_state.vector_store.get_all_documents()
    document_names = [doc["source"] for doc in docs_metadata]

    filter_option = st.selectbox(
        "Filtrar búsqueda por documento específico:",
        options=["Todos"] + document_names,
        index=0
    )
    filter_source = None if filter_option == "Todos" else filter_option

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Ver Fuentes"):
                    for src in msg["sources"]:
                        st.write(f"- 📄 **{src['source']}** (Pág. {src['page']})")

    if user_query := st.chat_input("Escribe tu pregunta académica aquí..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Buscando y generando respuesta..."):
                try:
                    result = st.session_state.rag_chain.ask(
                        query=user_query,
                        filter_source=filter_source
                    )
                    answer  = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("Ver Fuentes"):
                            for src in sources:
                                st.write(f"- 📄 **{src['source']}** (Pág. {src['page']})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    # Registra en Supabase (tolerante a fallos)
                    try:
                        st.session_state.db_manager.insert_consulta(
                            query             = user_query,
                            answer            = result["answer"],
                            sources           = result["sources"],
                            response_time_ms  = result["response_time_ms"],
                            similarity_scores = result["similarity_scores"]
                        )
                    except Exception as e:
                        print(f"[main] Error registrando consulta en Supabase: {e}")
                        # El usuario NO debe ver este error — es silencioso

                except Exception as e:
                    st.error(f"Ocurrió un error al procesar tu pregunta: {str(e)}")


def render_dashboard_tab():
    """
    Renderiza la pestaña de estadísticas.
    Estructura extendida con cuatro secciones:

    Sección 1: KPIs globales (st.metric)
               total docs, chunks, palabras, consultas

    Sección 2: Clustering ML
               distribución por cluster_label (st.bar_chart)
               silhouette_score como st.metric

    Sección 3: Estadísticas de uso
               consultas por día (st.line_chart)
               documentos más consultados (st.bar_chart)
               tiempo promedio de respuesta (st.metric)

    Sección 4: Tabla de documentos (st.dataframe)
               con title, author, cluster_label, chunks, fecha

    Input:  nada (lee de st.session_state)
    Output: nada

    Contiene:
      - st.metric() para totales globales
      - st.bar_chart() para distribución de chunks
      - st.dataframe() para tabla de documentos
    """
    st.header("📊 Métricas del Sistema RAG")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Global", "🤖 Clustering ML",
        "💬 Consultas", "📋 Documentos"
    ])

    with tab1:
        stats = st.session_state.dashboard.get_global_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Documentos",  stats.get("total_documents", 0))
        col2.metric("Chunks",      stats.get("total_chunks",    0))
        col3.metric("Páginas",     stats.get("total_pages",     0))

    with tab2:
        ml_stats = st.session_state.dashboard.get_ml_stats()
        if ml_stats["total_clusters"] == 0:
            st.info("Sube al menos 4 documentos para activar "
                    "la clasificación automática.")
        else:
            st.metric("Clusters detectados", ml_stats["total_clusters"])
            st.bar_chart(ml_stats["docs_per_cluster"])

    with tab3:
        usage = st.session_state.dashboard.get_usage_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total consultas",
                    usage.get("total_consultas", 0))
        col2.metric("Tiempo promedio (ms)",
                    usage.get("avg_response_time_ms", 0))
        col3.metric("Sin respuesta",
                    usage.get("consultas_sin_respuesta", 0))
        if usage.get("consultas_por_dia"):
            st.line_chart(usage["consultas_por_dia"])
        if usage.get("documentos_mas_consultados"):
            st.subheader("Documentos más consultados")
            st.dataframe(usage["documentos_mas_consultados"])

    with tab4:
        docs = st.session_state.db_manager.get_all_document_stats()
        if docs:
            st.dataframe(docs)
        else:
            st.info("No hay documentos indexados aún.")


def render_text_to_sql_tab():
    """
    Pestaña adicional para consultas en lenguaje natural
    sobre los datos de Supabase.

    Flujo:
    1. Usuario escribe pregunta sobre los datos
    2. rag_chain.text_to_sql(query, schema) → SQL
    3. Muestra el SQL generado al usuario (transparencia)
    4. db_manager.execute_readonly_query(sql) → resultados
    5. Muestra resultados en st.dataframe()
    6. rag_chain.generate_summary(resultados) → resumen en lenguaje natural
    """
    st.subheader("Consulta los datos en lenguaje natural")
    st.caption("Pregunta sobre estadísticas de documentos y consultas.")

    schema = """
    dim_documentos(id_documento, source, title, author, subject,
                   total_chunks, total_pages, total_words,
                   cluster_label, upload_date)
    fact_consultas(id_consulta, pregunta, response_time_ms, timestamp)
    dim_resultados(id_resultado, respuesta, avg_similarity, sin_respuesta)
    bridge_consulta_docs(id_consulta, id_documento)
    """

    query_sql = st.text_input(
        "¿Qué quieres saber sobre los datos?",
        placeholder="¿Cuántos documentos se subieron este mes?"
    )

    if query_sql:
        with st.spinner("Generando consulta SQL..."):
            try:
                sql = st.session_state.rag_chain.text_to_sql(
                    query_sql, schema
                )
                st.code(sql, language="sql")

                results = st.session_state.db_manager\
                            .execute_readonly_query(sql)

                if results:
                    st.dataframe(results)
                    summary = st.session_state.rag_chain.generate_summary([
                        {"text": str(results[:5]),
                         "metadata": {"source": "sql_result"}}
                    ])
                    st.info(f"💡 {summary}")
                else:
                    st.info("La consulta no retornó resultados.")

            except ValueError as e:
                st.error(f"Query no permitida: {e}")
            except Exception as e:
                st.error(f"Error ejecutando consulta: {e}")


def main():
    """
    Punto de entrada principal de la app Streamlit.

    Orden de ejecución:
      1. st.set_page_config()
      2. init_session_state()
      3. render_sidebar()
      4. st.tabs() → render_chat_tab() / render_text_to_sql_tab() / render_dashboard_tab()
    """
    st.set_page_config(
        page_title="RAG Académico",
        page_icon="📚",
        layout="wide"
    )
    init_session_state()
    render_sidebar()

    tab_chat, tab_sql, tab_dash = st.tabs([
        "💬 Chat", "🔍 Consulta SQL", "📊 Dashboard"
    ])

    with tab_chat:
        render_chat_tab()

    with tab_sql:
        render_text_to_sql_tab()

    with tab_dash:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
