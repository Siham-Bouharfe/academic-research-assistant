from llama_index.core import VectorStoreIndex

def build_retriever(index: VectorStoreIndex, top_k: int = 5):
    """
    Crée un retriever à partir de l'index.
    
    top_k = nombre de chunks retournés par requête
    Plus top_k est grand, plus le contexte est riche,
    mais plus le prompt sera long (coût en tokens).
    """

    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever


def retrieve(retriever, query: str):
    """
    Effectue une recherche sémantique dans l'index.
    
    Contrairement à une recherche par mots-clés (Google),
    la recherche sémantique comprend le SENS de la question.
    Ex: "impacts du réchauffement" trouvera aussi "effets du changement climatique"
    """

    print(f"\nRecherche pour : '{query}'")
    nodes = retriever.retrieve(query)

    print(f"{len(nodes)} passages trouvés :\n")
    for i, node in enumerate(nodes):
        print(f"--- Passage {i+1} (score: {node.score:.3f}) ---")
        print(node.text[:300])  # Affiche les 300 premiers caractères
        print()

    return nodes