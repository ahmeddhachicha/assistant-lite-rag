"""Appel au LLM via l'API locale d'Ollama."""

import requests

PROMPT_TEMPLATE = """Tu es un assistant documentaire. Voici des extraits d'un document :

{context}

Question : {question}

Consignes :
- Réponds en français, uniquement à partir des extraits ci-dessus.
- N'invente aucune information.
- Si les extraits contiennent des éléments de réponse, résume-les fidèlement.
- Si et seulement si les extraits ne contiennent aucun élément de réponse, réponds exactement : "Je ne trouve pas cette information dans le corpus."

Réponse :"""


class OllamaUnavailableError(Exception):
    """Levée quand le serveur Ollama est injoignable."""


def build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n---\n\n".join(passages)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate(question: str, passages: list[str], model: str, url: str, timeout: int) -> str:
    prompt = build_prompt(question, passages)
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # temperature 0 : réponses déterministes ; num_predict :
                # borne la génération (les très petits modèles peuvent boucler)
                "options": {"temperature": 0, "num_predict": 512},
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Ollama est injoignable à l'adresse {url}. "
            "Vérifiez que `ollama serve` est lancé."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(
            f"Ollama n'a pas répondu dans le délai imparti ({timeout} s). "
            "Réessayez ou choisissez un modèle plus léger."
        ) from exc
    return response.json()["response"].strip()
