# orchestrator.py

from agents.collector import create_collector_agent
from agents.analyst import create_analyst_agent
from agents.writer import create_writer_agent
from agents.validator import create_validator_agent

class ResearchOrchestrator:
    """
    L'orchestrateur coordonne les 4 agents dans l'ordre :
    Collecteur → Analyste → Rédacteur → Vérificateur
    """

    def __init__(self, retriever):
        print("🚀 Initialisation des agents...")
        self.collector = create_collector_agent(retriever)
        self.analyst = create_analyst_agent()
        self.writer = create_writer_agent()
        self.validator = create_validator_agent()
        print("✅ Les 4 agents sont prêts !")

    def run(self, research_question: str) -> dict:
        """
        Lance le pipeline complet pour une question de recherche.
        Retourne un dictionnaire avec les résultats de chaque agent.
        """

        print(f"\n{'='*60}")
        print(f"📚 QUESTION DE RECHERCHE : {research_question}")
        print(f"{'='*60}\n")

        # ÉTAPE 1 : Collecteur cherche les passages pertinents
        print("\n🔍 ÉTAPE 1/4 — Agent Collecteur...")
        collected = self.collector.invoke({
            "input": research_question
        })
        collected_text = collected["output"]
        print(f"✅ Collecte terminée.")

        # ÉTAPE 2 : Analyste analyse les passages collectés
        print("\n📊 ÉTAPE 2/4 — Agent Analyste...")
        analyzed = self.analyst.invoke({
            "input": f"""
            Question de recherche : {research_question}
            Passages collectés : {collected_text}
            Analyse ces passages et identifie les tendances,
            compare les approches et détecte les lacunes.
            """
        })
        analyzed_text = analyzed["output"]
        print(f"✅ Analyse terminée.")

        # ÉTAPE 3 : Rédacteur génère la synthèse
        print("\n✍️ ÉTAPE 3/4 — Agent Rédacteur...")
        written = self.writer.invoke({
            "input": f"""
            Question de recherche : {research_question}
            Analyse : {analyzed_text}
            Rédige une synthèse académique complète et structurée.
            """
        })
        written_text = written["output"]
        print(f"✅ Rédaction terminée.")

        # ÉTAPE 4 : Vérificateur valide la synthèse
        print("\n✔️ ÉTAPE 4/4 — Agent Vérificateur...")
        validated = self.validator.invoke({
            "input": f"""
            Synthèse à vérifier : {written_text}
            Vérifie la cohérence, les citations et la qualité.
            """
        })
        validated_text = validated["output"]
        print(f"✅ Validation terminée.")

        # Résultat final
        result = {
            "question": research_question,
            "collected_passages": collected_text,
            "analysis": analyzed_text,
            "synthesis": written_text,
            "validation": validated_text
        }

        return result