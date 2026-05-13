from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

import chromadb
import os


def build_index(papers_dir: str = "data/papers", persist_dir: str = "storage"):
    """
    Charge tous les PDFs du dossier, extrait correctement le texte,
    génère les embeddings et les stocke dans ChromaDB.
    """

    # --- ÉTAPE 1 : CHARGEMENT DES PDFS ---
    print("Chargement des documents PDF...")

    loader = PDFReader()
    documents = []

    for file in os.listdir(papers_dir):

        if file.endswith(".pdf"):

            path = os.path.join(papers_dir, file)

            print(f"Lecture de : {file}")

            pdf_docs = loader.load_data(file=path)

            documents.extend(pdf_docs)

    print(f"{len(documents)} pages chargées.")

    # --- ÉTAPE 2 : EMBEDDINGS ---
    print("Chargement du modèle d'embedding...")

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.embed_model = embed_model
    Settings.llm = None

    # --- ÉTAPE 3 : CHROMADB ---
    print("Connexion à ChromaDB...")

    chroma_client = chromadb.PersistentClient(path=persist_dir)

    chroma_collection = chroma_client.get_or_create_collection(
        "academic_papers"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # --- ÉTAPE 4 : INDEXATION ---
    print("Indexation en cours (chunking + embeddings)...")

    Settings.text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("Indexation terminée !")
    print(f"Index sauvegardé dans : {persist_dir}")

    return index


def load_index(persist_dir: str = "storage"):
    """
    Charge un index déjà existant depuis ChromaDB.
    """

    print("Chargement de l'index existant...")

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.embed_model = embed_model
    Settings.llm = None

    chroma_client = chromadb.PersistentClient(path=persist_dir)

    chroma_collection = chroma_client.get_or_create_collection(
        "academic_papers"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    print("Index chargé depuis ChromaDB.")

    return index