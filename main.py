# main.py

import os
from rag.ingestor import build_index, load_index
from rag.retriever import build_retriever
from orchestrator import ResearchOrchestrator

def main():
    print("🎓 Assistant de Recherche Académique")
    print("="*60)

    # Charger ou créer l'index RAG
    if os.path.exists("storage") and os.listdir("storage"):
        print("📦 Chargement de l'index existant...")
        index = load_index("storage")
    else:
        print("🆕 Création de l'index...")
        index = build_index("data/papers", "storage")

    # Créer le retriever
    retriever = build_retriever(index, top_k=3)

    # Créer l'orchestrateur
    orchestrator = ResearchOrchestrator(retriever)

    # Lancer une question de recherche
    question = "What are the main contributions and methodologies in these research papers?"

    result = orchestrator.run(question)

    # Afficher le résultat final
    print("\n" + "="*60)
    print("📄 SYNTHÈSE FINALE")
    print("="*60)
    print(result["synthesis"])
    print("\n" + "="*60)
    print("✔️ RAPPORT DE VALIDATION")
    print("="*60)
    print(result["validation"])

if __name__ == "__main__":
    main()