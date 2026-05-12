# test_collector.py
import os
from rag.ingestor import build_index, load_index
from rag.retriever import build_retriever
from agents.collector import create_collector_agent

# Charger le RAG
if os.path.exists("storage") and os.listdir("storage"):
    index = load_index("storage")
else:
    index = build_index("data/papers", "storage")

retriever = build_retriever(index, top_k=3)

# Créer l'agent
collector = create_collector_agent(retriever)

# Tester
result = collector.invoke({
    "input": "What are the main research contributions in these papers?"
})

print("\n✅ RÉSULTAT FINAL :")
print(result["output"])