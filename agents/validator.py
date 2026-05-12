# agents/validator.py

# from langchain.agents import create_react_agent
# from langchain.agents import AgentExecutor
# from langchain.tools import Tool
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama

# agents/validator.py

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def create_validator_agent():

    llm = ChatOllama(model="gemma3:4b", temperature=0)

    def run(synthesis: str) -> str:
        messages = [
            SystemMessage(content="""Tu es un Agent Vérificateur expert 
            en qualité académique. Tu vérifies la cohérence, 
            les citations et la qualité globale des synthèses."""),
            HumanMessage(content=f"""
            Synthèse à vérifier : {synthesis}
            
            Vérifie et évalue :
            1. Cohérence logique du contenu
            2. Qualité académique
            3. Structure et clarté
            
            Donne un score /10 et liste les points à améliorer.
            """)
        ]

        response = llm.invoke(messages)
        return response.content

    return run