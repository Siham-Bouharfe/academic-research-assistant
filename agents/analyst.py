# agents/analyst.py

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def create_analyst_agent():

    llm = ChatOllama(model="gemma3:4b", temperature=0)

    def run(collected_text: str, question: str) -> str:
        messages = [
            SystemMessage(content="""Tu es un Agent Analyste expert 
            en recherche académique. Tu analyses les passages collectés 
            pour identifier les tendances, comparer les approches 
            et détecter les lacunes."""),
            HumanMessage(content=f"""
            Question de recherche : {question}
            
            Passages collectés : {collected_text}
            
            Effectue une analyse complète :
            1. Tendances principales
            2. Comparaison des approches méthodologiques
            3. Lacunes détectées dans la littérature
            """)
        ]

        response = llm.invoke(messages)
        return response.content

    return run