"""Tests demandés par le sujet.

1. Trois questions : deux présentes dans le corpus, une absente.
2. Comparaison de deux configurations (nombre de segments récupérés).

Résultats écrits dans tests/resultats_tests.md.
Lancement : python -m tests.run_tests (Ollama doit être démarré).
"""

from rag import config
from rag.pipeline import RagPipeline

QUESTIONS = [
    ("Que dit le discours sur le changement climatique ?", "présente"),
    ("Que dit le discours sur Medicare et la Sécurité sociale ?", "présente"),
    ("Quel est le prix du bitcoin mentionné dans le discours ?", "absente"),
]

CONFIGS = [
    {"label": "Config A", "chunk_size": 800, "top_k": 3},
    {"label": "Config B", "chunk_size": 800, "top_k": 6},
]


def run() -> None:
    lines = [
        "# Résultats des tests\n",
        f"Modèle : `{config.OLLAMA_MODEL}` — corpus : `corpus_de_travail.txt`\n",
        "## Comparaison des configurations\n",
        "Paramètre modifié entre les deux configurations : "
        "**nombre de segments récupérés** (3 vs 6), taille des segments fixe (800).\n",
        "| Config | Question | Réponse attendue | Statut | Segments | Temps (s) | Réponse (extrait) |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in CONFIGS:
        pipeline = RagPipeline(chunk_size=cfg["chunk_size"])
        for question, expected in QUESTIONS:
            result = pipeline.ask(question, top_k=cfg["top_k"])
            extract = result["answer"].replace("\n", " ")[:120]
            lines.append(
                f"| {cfg['label']} (k={cfg['top_k']}) | {question} | {expected} "
                f"| {result['status']} | {result['n_segments']} "
                f"| {result['elapsed_seconds']} | {extract}… |"
            )
            print(f"[{cfg['label']}] {question} -> {result['status']} "
                  f"({result['elapsed_seconds']} s)")

    with open("tests/resultats_tests.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nRésultats écrits dans tests/resultats_tests.md")


if __name__ == "__main__":
    run()
