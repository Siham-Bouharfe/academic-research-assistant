import os
from rag.ingestor import build_index, load_index
from rag.retriever import build_retriever, retrieve

PAPERS_DIR = "data/papers"
STORAGE_DIR = "storage"

# Si l'index existe déjà, on le charge. Sinon on le crée.
if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
    print("Index trouvé, chargement...")
    index = load_index(STORAGE_DIR)
else:
    print("Pas d'index trouvé, création...")
    index = build_index(PAPERS_DIR, STORAGE_DIR)

# Créer le retriever
retriever = build_retriever(index, top_k=3)

# Tester avec une vraie question
query = "What are the main contributions of this research?"
results = retrieve(retriever, query)