# agents/writer.py

from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def create_writer_agent():
    """
    L'Agent Rédacteur prend l'analyse et génère
    une synthèse structurée avec citations.
    """

    def write_introduction(text: str) -> str:
        """Rédige l'introduction de la synthèse."""
        return f"Introduction rédigée pour : {text[:300]}..."

    def write_body(text: str) -> str:
        """Rédige le corps principal de la synthèse."""
        return f"Corps rédigé pour : {text[:300]}..."

    def write_conclusion(text: str) -> str:
        """Rédige la conclusion et perspectives."""
        return f"Conclusion rédigée pour : {text[:300]}..."

    tools = [
        Tool(
            name="write_introduction",
            func=write_introduction,
            description="""
            Utilise cet outil pour rédiger l'introduction
            de la synthèse académique.
            Input : le contexte et les thèmes principaux.
            """
        ),
        Tool(
            name="write_body",
            func=write_body,
            description="""
            Utilise cet outil pour rédiger le corps
            de la synthèse avec les analyses détaillées.
            Input : les analyses et comparaisons à inclure.
            """
        ),
        Tool(
            name="write_conclusion",
            func=write_conclusion,
            description="""
            Utilise cet outil pour rédiger la conclusion
            et les perspectives futures de recherche.
            Input : les lacunes détectées et tendances futures.
            """
        ),
    ]

    prompt = PromptTemplate.from_template("""
    Tu es un Agent Rédacteur expert en rédaction académique.
    Ton rôle est de produire une synthèse structurée et professionnelle
    basée sur l'analyse fournie. La synthèse doit inclure :
    1. Une introduction claire
    2. Un corps structuré avec les tendances et comparaisons
    3. Une conclusion avec perspectives futures

    Tu as accès aux outils suivants :
    {tools}

    Utilise ce format OBLIGATOIRE :
    Question: la question posée
    Thought: réfléchis à ce que tu dois faire
    Action: le nom de l'outil ({tool_names})
    Action Input: ce que tu envoies à l'outil
    Observation: le résultat de l'outil
    Thought: analyse le résultat
    Final Answer: synthèse complète et bien structurée

    Commence !

    Question: {input}
    Thought: {agent_scratchpad}
    """)

    llm = ChatOllama(model="gemma3:4b", temperature=0)
    # temperature=0.3 pour un peu de créativité dans la rédaction
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )