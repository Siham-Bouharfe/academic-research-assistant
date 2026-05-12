from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import os

def build_index(papers_dir: str = "data/papers", persist_dir: str = "storage"):
    """
    Charge tous les PDFs du dossier, les découpe, génère les embeddings
    et les stocke dans ChromaDB.
    """

    # --- ÉTAPE 1 : INGESTION ---
    # SimpleDirectoryReader lit tous les fichiers du dossier (PDF, TXT, etc.)
    # Il extrait le texte brut de chaque document
    print("Chargement des documents...")
    documents = SimpleDirectoryReader(papers_dir).load_data()
    print(f"{len(documents)} documents chargés.")

    # --- ÉTAPE 2 : EMBEDDING ---
    # On utilise un modèle d'embedding GRATUIT de HuggingFace
    # Ce modèle transforme chaque chunk de texte en un vecteur numérique
    # (une liste de nombres qui représente le "sens" du texte)
    print("Chargement du modèle d'embedding...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # Ce modèle est léger (80MB) et très efficace pour l'anglais
        # Si vos articles sont en français, utilisez :
        # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    Settings.embed_model = embed_model
    Settings.llm = None  # On ne configure pas le LLM ici, seulement l'embedding

    # --- ÉTAPE 3 : VECTOR STORE (ChromaDB) ---
    # ChromaDB est une base de données qui stocke les vecteurs sur disque
    # Ainsi, on n'a pas besoin de re-indexer à chaque démarrage
    print("Connexion à ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("academic_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- ÉTAPE 4 : INDEXATION (CHUNKING + STOCKAGE) ---
    # VectorStoreIndex va automatiquement :
    # 1. Découper les documents en chunks (morceaux de ~512 tokens par défaut)
    # 2. Générer un embedding pour chaque chunk
    # 3. Stocker les embeddings dans ChromaDB
    print("Indexation en cours (chunking + embeddings)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True  # Affiche une barre de progression
    )
    print("Indexation terminée ! Index sauvegardé dans ./storage")

    return index


def load_index(persist_dir: str = "storage"):
    """
    Charge un index déjà existant depuis ChromaDB.
    À utiliser après la première indexation pour éviter de re-indexer.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model
    Settings.llm = None

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("academic_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    print("Index chargé depuis ChromaDB.")
    return index