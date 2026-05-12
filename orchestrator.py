# orchestrator.py

# from agents.collector import create_collector_agent
# from agents.analyst import create_analyst_agent
# from agents.writer import create_writer_agent
# from agents.validator import create_validator_agent

# orchestrator.py

from agents.collector import create_collector_agent
from agents.analyst import create_analyst_agent
from agents.writer import create_writer_agent
from agents.validator import create_validator_agent

class ResearchOrchestrator:

    def __init__(self, retriever):
        print("🚀 Initialisation des agents...")
        self.collector = create_collector_agent(retriever)
        self.analyst = create_analyst_agent()
        self.writer = create_writer_agent()
        self.validator = create_validator_agent()
        print("✅ Les 4 agents sont prêts !")

    def run(self, research_question: str) -> dict:

        print(f"\n{'='*60}")
        print(f"📚 QUESTION : {research_question}")
        print(f"{'='*60}\n")

        print("🔍 ÉTAPE 1/4 — Agent Collecteur...")
        collected = self.collector(research_question)
        print("✅ Collecte terminée.\n")

        print("📊 ÉTAPE 2/4 — Agent Analyste...")
        analyzed = self.analyst(collected, research_question)
        print("✅ Analyse terminée.\n")

        print("✍️ ÉTAPE 3/4 — Agent Rédacteur...")
        written = self.writer(analyzed, research_question)
        print("✅ Rédaction terminée.\n")

        print("✔️ ÉTAPE 4/4 — Agent Vérificateur...")
        validated = self.validator(written)
        print("✅ Validation terminée.\n")

        return {
            "question": research_question,
            "collected": collected,
            "analysis": analyzed,
            "synthesis": written,
            "validation": validated
        }