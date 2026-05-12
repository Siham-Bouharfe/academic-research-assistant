# agents/collector.py

# from langchain.agents import create_react_agent
# from langchain.agents import AgentExecutor
# from langchain.tools import Tool
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama  # ou ChatAnthropic, ChatOpenAI...

# agents/collector.py

from pyexpat.errors import messages

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def create_collector_agent(retriever):
    """
    Dans LangChain 1.x, on utilise directement le LLM
    avec des messages structurés au lieu de AgentExecutor.
    """

    llm = ChatOllama(model="gemma3:4b", temperature=0)

    def run(question: str) -> str:
        # Étape 1 : Récupérer les passages via RAG
        nodes = retriever.retrieve(question)

        if not nodes:
            context = "Aucun passage pertinent trouvé."
        else:
            context = "\n---\n".join([
                f"Passage {i+1} (score: {node.score:.2f}):\n{node.text}"
                for i, node in enumerate(nodes)
            ])

        # Étape 2 : Envoyer au LLM avec le contexte
        messages = [
            SystemMessage(content="""Tu es un Agent Collecteur spécialisé 
            en recherche académique. Ton rôle est d'extraire et résumer 
            les passages les plus pertinents des articles scientifiques."""),
            HumanMessage(content=f"""
            Question de recherche : {question}
            
            Passages trouvés dans les articles :
            {context}
            
            Résume les passages les plus pertinents pour répondre 
            à la question de recherche.
            """)
        ]
        print(type(llm))
        print(messages)
        response = llm.invoke(messages)
        return response.content

    return run