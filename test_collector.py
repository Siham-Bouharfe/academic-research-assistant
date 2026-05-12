# test_collector.py

import os
from rag.ingestor import build_index, load_index
from rag.retriever import build_retriever
from agents.collector import create_collector_agent

if os.path.exists("storage") and os.listdir("storage"):
    index = load_index("storage")
else:
    index = build_index("data/papers", "storage")

retriever = build_retriever(index, top_k=3)
collector = create_collector_agent(retriever)

result = collector("What are the main contributions of this research?")

print("\n✅ RÉSULTAT :")
print(result)