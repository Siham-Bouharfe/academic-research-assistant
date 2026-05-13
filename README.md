# Academic Research Assistant
Système multi-agents intelligent pour l'analyse de corpus scientifiques.

## Description
Ce projet implémente un système multi-agents basé sur LangChain et LlamaIndex
pour analyser automatiquement des articles scientifiques et produire
un état de l'art structuré.

## Architecture
Question de recherche
↓
[Agent Collecteur] → cherche dans les PDFs via RAG
↓
[Agent Analyste] → identifie tendances et lacunes
↓
[Agent Rédacteur] → génère la synthèse structurée
↓
[Agent Vérificateur] → valide la qualité finale

## Stack Technologique
 LangChain 1.2.x -> Développement des agents 
 LlamaIndex -> Pipeline RAG 
 Ollama (gemma3:4b) -> LLM local gratuit 
 ChromaDB -> Vector store 
 HuggingFace -> Modèle d'embedding 

 ## Structure du Projet
 academic-research-assistant/
├── data/
│   └── papers/          # Vos articles PDF ici
├── storage/             # Index ChromaDB (généré automatiquement)
├── rag/
│   ├── ingestor.py      # Chargement et indexation des PDFs
│   └── retriever.py     # Recherche sémantique
├── agents/
│   ├── collector.py     # Agent Collecteur (RAG)
│   ├── analyst.py       # Agent Analyste
│   ├── writer.py        # Agent Rédacteur
│   └── validator.py     # Agent Vérificateur
├── orchestrator.py      # Coordination des agents
├── main.py              # Point d'entrée
├── requirements.txt     # Dépendances
├── .env.example         # Template variables d'environnement
└── README.md            # Ce fichier

## Pipeline RAG
1. **Ingestion** : chargement des PDFs depuis `data/papers/`
2. **Chunking** : découpage en morceaux de 512 tokens
3. **Embedding** : vectorisation via `all-MiniLM-L6-v2`
4. **Stockage** : sauvegarde dans ChromaDB
5. **Retrieval** : recherche sémantique top-k

## Auteurs
- Siham BOUHARFE
- Amal Igoulalen