# agents/collector.py

from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama  # ou ChatAnthropic, ChatOpenAI...

def create_collector_agent(retriever):
    """
    Crée l'agent Collecteur.
    Cet agent reçoit une question et cherche les passages
    pertinents dans les PDFs via le RAG.
    """

    # --- ÉTAPE 1 : DÉFINIR L'OUTIL DE RECHERCHE ---
    # Un "tool" en LangChain c'est une fonction que l'agent
    # peut décider d'appeler ou non selon la situation
    def search_documents(query: str) -> str:
        """Recherche des passages pertinents dans les articles scientifiques."""
        nodes = retriever.retrieve(query)

        if not nodes:
            return "Aucun passage pertinent trouvé."

        # On formate les résultats en texte lisible
        results = []
        for i, node in enumerate(nodes):
            results.append(
                f"Passage {i+1} (pertinence: {node.score:.2f}):\n{node.text}\n"
            )
        return "\n---\n".join(results)

    # On emballe la fonction dans un objet Tool LangChain
    search_tool = Tool(
        name="search_academic_papers",
        # Ce nom est important : l'agent l'utilise pour
        # décider quel outil appeler
        func=search_documents,
        description="""
        Utilise cet outil pour rechercher des informations
        dans le corpus d'articles scientifiques.
        Input : une question ou un sujet de recherche en anglais.
        Output : les passages les plus pertinents trouvés.
        """
    )

    tools = [search_tool]

    # --- ÉTAPE 2 : DÉFINIR LE PROMPT (PERSONA) ---
    # C'est la "personnalité" et les instructions de l'agent
    # ReAct = Reasoning + Acting : l'agent réfléchit avant d'agir
    prompt = PromptTemplate.from_template("""
    Tu es un Agent Collecteur spécialisé dans la recherche académique.
    Ton rôle est de trouver les passages les plus pertinents dans
    un corpus d'articles scientifiques pour répondre à une question.

    Tu as accès aux outils suivants :
    {tools}

    Pour répondre, utilise ce format OBLIGATOIRE :
    Question: la question posée
    Thought: réfléchis à ce que tu dois faire
    Action: le nom de l'outil à utiliser ({tool_names})
    Action Input: ce que tu envoies à l'outil
    Observation: le résultat de l'outil
    Thought: analyse le résultat
    Final Answer: résumé des passages trouvés, bien structuré

    Commence !

    Question: {input}
    Thought: {agent_scratchpad}
    """)

    # --- ÉTAPE 3 : DÉFINIR LE LLM ---
    # C'est le cerveau de l'agent
    # On utilise Ollama (gratuit et local)
    llm = ChatOllama(model="gemma3:4b", temperature=0)

    # --- ÉTAPE 4 : ASSEMBLER L'AGENT ---
    # create_react_agent combine le LLM + le prompt + les tools
    # pour créer un agent capable de raisonner
    agent = create_react_agent(llm, tools, prompt)

    # AgentExecutor est le "moteur" qui fait tourner l'agent
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,     # Affiche le raisonnement de l'agent
        max_iterations=3  # Évite les boucles infinies
    )

    return agent_executor