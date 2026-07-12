# 📚 RAG de Archivos Académicos

Este proyecto es una aplicación web interactiva desarrollada con **Streamlit** que implementa un sistema RAG (Retrieval-Augmented Generation) para consultar y extraer información precisa de documentos académicos en formato PDF. El sistema utiliza **ChromaDB** como base de datos vectorial local, **Sentence-Transformers** para la generación de embeddings semánticos y la API de **Groq** (LLM) para generar respuestas basadas en el contexto recuperado de los documentos.

---

## 🛠️ Tecnologías y Librerías

El backend y procesamiento de datos está construido sobre el siguiente stack tecnológico:
* **Streamlit**: Para la interfaz de usuario interactiva y el dashboard.
* **PyMuPDF (fitz)**: Para la lectura y extracción de texto de documentos PDF.
* **Sentence-Transformers (MiniLM-L12)**: Generación local de embeddings semánticos multilingües.
* **ChromaDB**: Base de datos vectorial persistente para almacenar y consultar los fragmentos indexados.
* **Groq API**: Cliente para interactuar con modelos de lenguaje masivo (LLM) de baja latencia (Llama 3.1).
* **Python-dotenv**: Gestión de variables de entorno de configuración.

---

## 📋 Requisitos Previos

* **Python 3.10 o superior** instalado en el sistema.
* Una clave API activa de **Groq** (consíguela gratis en [Groq Console](https://console.groq.com/)).

---

## 🚀 Instalación y Configuración

Sigue estos pasos para clonar, configurar e iniciar el proyecto en tu entorno local:

### 1. Clonar el repositorio
```bash
git clone https://github.com/dasantimate14/GI-RAG-ACADEMICO.git
cd Parcial2-RAG_Archivos_Academicos
```

### 2. Crear y activar un entorno virtual (Recomendado)
En Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
En macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias y librerías
Instala todas las librerías necesarias especificadas en el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno (`.env`)
Crea un archivo llamado `.env` en la raíz del proyecto y agrega tu API Key de Groq y las rutas opcionales de almacenamiento:
```env
GROQ_API_KEY=tu_groq_api_key_aqui
CHROMA_PATH=./data/chroma_db
UPLOAD_PATH=./data/uploads
```

---

## 💻 Ejecución de la Aplicación

Para iniciar el servidor de desarrollo de Streamlit, ejecuta el siguiente comando en la raíz del proyecto con tu entorno virtual activo:

```bash
streamlit run main.py
```

Una vez que se inicie, abre tu navegador e ingresa a `http://localhost:8501`.

---

## 📁 Estructura del Proyecto

```text
├── app/
│   ├── __init__.py
│   ├── dashboard.py         # Lógica de procesamiento de estadísticas y gráficos
│   ├── embeddings.py        # Generador de embeddings semánticos multilingües
│   ├── pdf_processor.py     # Extracción, limpieza y segmentación (chunking) de PDFs
│   ├── rag_chain.py         # Orquestación del flujo RAG con Groq
│   └── vector_store.py      # Operaciones con ChromaDB (indexar, buscar, borrar)
├── data/                    # Almacenamiento local de ChromaDB y PDFs subidos
├── config.py                # Variables globales y configuración centralizada
├── main.py                  # Interfaz de usuario en Streamlit (Chatbot y Dashboard)
├── requirements.txt         # Lista de dependencias del proyecto
└── README.md                # Documentación del proyecto
```

---

## 💡 Características Clave

1. **Gestión Completa de PDFs**: Sube múltiples documentos desde el sidebar. El sistema los procesará en fragmentos (chunks) con solapamiento (*overlap*) configurable y los guardará en la base de datos vectorial local.
2. **Chatbot con Citaciones**: Puedes preguntar sobre tus documentos e indicarle si deseas filtrar la búsqueda para un solo documento en específico. El chatbot responderá citando los archivos fuente y páginas de donde obtuvo los hechos.
3. **Dashboard de Estadísticas**: Vista con métricas globales del sistema, gráfico de distribución de chunks por cada archivo y una tabla detallada de los documentos indexados en el sistema.
