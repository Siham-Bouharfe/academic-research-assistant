# demo.py

import os
from rag.ingestor import load_index
from rag.retriever import build_retriever
from agents.collector import create_collector_agent
from agents.analyst import create_analyst_agent
from agents.writer import create_writer_agent
from agents.validator import create_validator_agent

question = "What are the main contributions and methodologies in these papers?"

# Charger le RAG
index = load_index("storage")
retriever = build_retriever(index, top_k=3)

# --- AGENT 1 : Collecteur ---
print("\n" + "="*60)
print("AGENT 1 — COLLECTEUR")
print("="*60)
collector = create_collector_agent(retriever)
collected = collector(question)
print(collected)
input("\nAppuyez sur Entrée pour continuer...")

# --- AGENT 2 : Analyste ---
print("\n" + "="*60)
print("AGENT 2 — ANALYSTE")
print("="*60)
analyst = create_analyst_agent()
analyzed = analyst(collected, question)
print(analyzed)
input("\nAppuyez sur Entrée pour continuer...")

# --- AGENT 3 : Rédacteur ---
print("\n" + "="*60)
print("AGENT 3 — RÉDACTEUR")
print("="*60)
writer = create_writer_agent()
written = writer(analyzed, question)
print(written)
input("\nAppuyez sur Entrée pour continuer...")

# --- AGENT 4 : Vérificateur ---
print("\n" + "="*60)
print("AGENT 4 — VÉRIFICATEUR")
print("="*60)
validator = create_validator_agent()
validated = validator(written)
print(validated)