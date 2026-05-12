# agents/validator.py

from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def create_validator_agent():
    """
    L'Agent Vérificateur contrôle la qualité de la synthèse :
    cohérence, citations, structure et qualité académique.
    """

    def check_coherence(text: str) -> str:
        """Vérifie la cohérence logique du texte."""
        return f"Vérification cohérence : {text[:300]}..."

    def check_citations(text: str) -> str:
        """Vérifie que les citations sont correctes."""
        return f"Vérification citations : {text[:300]}..."

    def check_quality(text: str) -> str:
        """Évalue la qualité académique globale."""
        return f"Évaluation qualité : {text[:300]}..."

    tools = [
        Tool(
            name="check_coherence",
            func=check_coherence,
            description="""
            Vérifie la cohérence logique et la structure
            de la synthèse rédigée.
            Input : le texte de la synthèse à vérifier.
            """
        ),
        Tool(
            name="check_citations",
            func=check_citations,
            description="""
            Vérifie que les citations et références
            sont correctes et bien formatées.
            Input : le texte contenant les citations.
            """
        ),
        Tool(
            name="check_quality",
            func=check_quality,
            description="""
            Évalue la qualité académique globale
            et suggère des améliorations si nécessaire.
            Input : la synthèse complète à évaluer.
            """
        ),
    ]

    prompt = PromptTemplate.from_template("""
    Tu es un Agent Vérificateur expert en qualité académique.
    Ton rôle est de vérifier la synthèse produite et de t'assurer :
    1. La cohérence logique du contenu
    2. La correction des citations et références
    3. La qualité académique globale
    Si tu trouves des problèmes, indique-les clairement.
    Si tout est correct, valide la synthèse.

    Tu as accès aux outils suivants :
    {tools}

    Utilise ce format OBLIGATOIRE :
    Question: la question posée
    Thought: réfléchis à ce que tu dois faire
    Action: le nom de l'outil ({tool_names})
    Action Input: ce que tu envoies à l'outil
    Observation: le résultat de l'outil
    Thought: analyse le résultat
    Final Answer: rapport de validation avec score /10

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