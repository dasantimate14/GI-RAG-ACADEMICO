import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Base directory (rag_academico)
BASE_DIR = Path(__file__).resolve().parent

# Project root (where .env resides)
PROJECT_ROOT = BASE_DIR.parent

# Carga las variables del archivo .env al entorno buscando en el directorio raíz
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Asegurar que el directorio base y el de la app estén en el PYTHONPATH para ejecuciones directas de scripts
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================
# RUTAS DE ALMACENAMIENTO
# ============================================
# Usamos rutas absolutas basadas en la ubicación del proyecto para evitar errores de ejecución
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma_db"))
UPLOAD_PATH = os.getenv("UPLOAD_PATH", str(BASE_DIR / "data" / "uploads"))
COLLECTION_NAME = "documentos_academicos"

# Crear directorios si no existen
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_PATH).mkdir(parents=True, exist_ok=True)

# ============================================
# CHUNKING
# ============================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================
# EMBEDDINGS
# ============================================
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ============================================
# BÚSQUEDA / RETRIEVAL
# ============================================
TOP_K_RESULTS = 3

# ============================================
# LLM
# ============================================
LLM_PROVIDER ="groq"
LLM_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY  is None:
    raise ValueError("GROQ_API_KEY environment variable not set")