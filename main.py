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
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DBManager()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = RAGChain(st.session_state.vector_store)
    if "ml_classifier" not in st.session_state:
        st.session_state.ml_classifier = MLClassifier(st.session_state.vector_store, st.session_state.rag_chain)
    if "metadata_extractor" not in st.session_state:
        st.session_state.metadata_extractor = MetadataExtractor(st.session_state.rag_chain)
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = Dashboard(st.session_state.vector_store)
    if "messages" not in st.session_state:
        st.session_state.messages = []


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
            pdf_processor = PDFProcessor()
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name

                if st.session_state.vector_store.document_exists(filename):
                    st.info(f"El archivo '{filename}' ya está indexado.")
                else:
                    with st.spinner(f"Indexando {filename}..."):
                        try:
                            chunks = pdf_processor.process(uploaded_file)
                            indexed_count = st.session_state.vector_store.add_documents(chunks)
                            st.success(f"¡Indexado exitosamente! ({indexed_count} chunks)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al procesar '{filename}': {str(e)}")

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
                    response_dict = st.session_state.rag_chain.ask(
                        query=user_query,
                        filter_source=filter_source
                    )
                    answer = response_dict["answer"]
                    sources = response_dict["sources"]

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

    dashboard = st.session_state.dashboard
    stats = dashboard.get_global_stats()

    if stats.get("total_documents", 0) == 0:
        st.info("No hay documentos en la base de datos. Sube archivos PDF desde el panel lateral para ver estadísticas.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documentos", stats.get("total_documents", 0))
    with col2:
        st.metric("Total Chunks (Fragmentos)", stats.get("total_chunks", 0))
    with col3:
        st.metric("Total Páginas", stats.get("total_pages", 0))

    st.markdown("---")

    st.subheader("📈 Distribución de Chunks por Documento")
    chart_data = dashboard.get_chunks_chart_data()
    if chart_data:
        st.bar_chart(chart_data)

    st.markdown("---")

    st.subheader("📋 Detalle de Documentos Indexados")
    docs_table = dashboard.get_documents_table()
    if docs_table:
        st.dataframe(docs_table, use_container_width=True)

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

def main():
    """
    Punto de entrada principal de la app Streamlit.

    Orden de ejecución:
      1. st.set_page_config()
      2. init_session_state()
      3. render_sidebar()
      4. st.tabs() → render_chat_tab() / render_dashboard_tab()
    """
    st.set_page_config(
        page_title="RAG Académico",
        page_icon="📚",
        layout="wide"
    )
    init_session_state()
    render_sidebar()

    tab_chat, tab_dashboard = st.tabs(["💬 Chatbot", "📊 Dashboard"])

    with tab_chat:
        render_chat_tab()

    with tab_dashboard:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
