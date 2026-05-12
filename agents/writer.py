# agents/writer.py

# from langchain.agents import create_react_agent
# from langchain.agents import AgentExecutor
# from langchain.tools import Tool
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama

# agents/writer.py

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def create_writer_agent():

    llm = ChatOllama(model="gemma3:4b", temperature=0.3)

    def run(analysis: str, question: str) -> str:
        messages = [
            SystemMessage(content="""Tu es un Agent Rédacteur expert 
            en rédaction académique. Tu produis des synthèses 
            structurées et professionnelles."""),
            HumanMessage(content=f"""
            Question de recherche : {question}
            
            Analyse fournie : {analysis}
            
            Rédige une synthèse académique complète avec :
            1. Introduction
            2. Corps structuré (tendances, comparaisons)
            3. Conclusion et perspectives futures
            """)
        ]

        response = llm.invoke(messages)
        return response.content

    return run