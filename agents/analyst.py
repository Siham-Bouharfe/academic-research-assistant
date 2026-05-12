# agents/analyst.py

from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def create_analyst_agent():
    """
    L'Agent Analyste reçoit les passages trouvés par le Collecteur
    et les analyse pour identifier les tendances, comparer les 
    approches et détecter les lacunes de la recherche.
    """

    # --- OUTIL 1 : Analyser les tendances ---
    def analyze_trends(text: str) -> str:
        """Identifie les tendances principales dans les passages."""
        # L'agent lui-même va raisonner sur le texte
        # cet outil sert juste à structurer l'entrée
        return f"Analyse des tendances dans : {text[:500]}..."

    # --- OUTIL 2 : Comparer les approches ---
    def compare_approaches(text: str) -> str:
        """Compare les différentes approches méthodologiques."""
        return f"Comparaison des approches dans : {text[:500]}..."

    # --- OUTIL 3 : Détecter les lacunes ---
    def detect_gaps(text: str) -> str:
        """Détecte les lacunes et manques dans la littérature."""
        return f"Détection des lacunes dans : {text[:500]}..."

    tools = [
        Tool(
            name="analyze_trends",
            func=analyze_trends,
            description="""
            Utilise cet outil pour identifier les tendances 
            principales dans les passages collectés.
            Input : les passages de recherche à analyser.
            """
        ),
        Tool(
            name="compare_approaches",
            func=compare_approaches,
            description="""
            Utilise cet outil pour comparer les différentes 
            approches et méthodologies des articles.
            Input : les passages décrivant les méthodologies.
            """
        ),
        Tool(
            name="detect_gaps",
            func=detect_gaps,
            description="""
            Utilise cet outil pour identifier les lacunes
            et questions non résolues dans la littérature.
            Input : les passages à analyser pour les lacunes.
            """
        ),
    ]

    prompt = PromptTemplate.from_template("""
    Tu es un Agent Analyste expert en recherche académique.
    Ton rôle est d'analyser les passages collectés et de :
    1. Identifier les tendances principales
    2. Comparer les approches méthodologiques
    3. Détecter les lacunes dans la littérature

    Tu as accès aux outils suivants :
    {tools}

    Utilise ce format OBLIGATOIRE :
    Question: la question posée
    Thought: réfléchis à ce que tu dois faire
    Action: le nom de l'outil ({tool_names})
    Action Input: ce que tu envoies à l'outil
    Observation: le résultat de l'outil
    Thought: analyse le résultat
    Final Answer: analyse complète et structurée

    Commence !

    Question: {input}
    Thought: {agent_scratchpad}
    """)

    llm = ChatOllama(model="gemma3:4b", temperature=0)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )