"""Webapp Streamlit — Assistant documentaire Lite RAG."""

import os

import streamlit as st

# Streamlit Community Cloud : recopie les secrets (OLLAMA_URL, etc.) dans
# l'environnement avant le chargement de la configuration
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from rag import config
from rag.llm import OllamaUnavailableError
from rag.pipeline import RagPipeline

st.set_page_config(
    page_title="Assistant documentaire Lite RAG",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 48rem; padding-top: 1.5rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .app-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        color: #8e8ea0;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.4rem 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: #f2f4f8;
        border-radius: 1rem;
        padding: 0.4rem 0.9rem;
    }
    .meta-line {
        color: #8e8ea0;
        font-size: 0.78rem;
        margin-top: 0.5rem;
    }
    .passage-card {
        background: #f7f7f8;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #40414f;
    }
    .passage-meta {
        font-size: 0.72rem;
        color: #8e8ea0;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Indexation du corpus…")
def get_pipeline() -> RagPipeline:
    return RagPipeline()


pipeline_ready = True
try:
    pipeline = get_pipeline()
except Exception as exc:  # corpus manquant, etc.
    pipeline_ready = False
    st.error(f"Impossible d'initialiser le pipeline : {exc}")

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown(
        f"**Modèle LLM (Ollama)**\n\n`{config.OLLAMA_MODEL}`\n\n"
        f"**Modèle d'embedding**\n\n`all-MiniLM-L6-v2` (ChromaDB)"
    )
    st.divider()
    st.markdown("## Corpus")
    if pipeline_ready:
        st.markdown(
            f"- Fichier : `{pipeline.corpus_path}`\n"
            f"- Segments indexés : **{pipeline.n_chunks}**\n"
            f"- Taille de segment : **{pipeline.chunk_size}** caractères\n"
            f"- Segments récupérés (top-k) : **{config.TOP_K}**"
        )
    st.divider()
    st.caption(
        "Paramètres modifiables via le fichier `.env` "
        "(voir `.env.example`). Journal des requêtes : "
        f"`{config.LOG_FILE}`."
    )

st.markdown('<div class="app-title">Assistant documentaire Lite RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Posez une question sur le corpus '
    "(discours d'investiture présidentielle, 2013) — les réponses sont "
    "fondées uniquement sur les passages retrouvés.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_assistant(message: dict) -> None:
    st.write(message["content"])
    if message.get("meta"):
        st.markdown(
            f'<div class="meta-line">{message["meta"]}</div>',
            unsafe_allow_html=True,
        )
    if message.get("passages"):
        with st.expander(f"Passages utilisés ({len(message['passages'])})"):
            for i, passage in enumerate(message["passages"], start=1):
                st.markdown(
                    f'<div class="passage-card">'
                    f'<div class="passage-meta">Passage {i} — distance : '
                    f'{passage["distance"]}</div>{passage["text"]}</div>',
                    unsafe_allow_html=True,
                )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.write(message["content"])

question = st.chat_input("Posez votre question…", disabled=not pipeline_ready)

if question is not None and pipeline_ready:
    if not question.strip():
        st.warning("Veuillez saisir une question avant de lancer la recherche.")
    else:
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Recherche dans le corpus…"):
                    result = pipeline.ask(question, top_k=config.TOP_K)
            except OllamaUnavailableError as exc:
                message = {"role": "assistant", "content": str(exc)}
                st.session_state.messages.append(message)
                render_assistant(message)
            except Exception as exc:
                message = {"role": "assistant", "content": f"Erreur inattendue : {exc}"}
                st.session_state.messages.append(message)
                render_assistant(message)
            else:
                message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": (
                        f'{config.OLLAMA_MODEL} · {result["elapsed_seconds"]} s · '
                        f'{result["n_segments"]} segments récupérés'
                    ),
                    "passages": result["passages"],
                }
                st.session_state.messages.append(message)
                render_assistant(message)
