"""Webapp Streamlit — Assistant documentaire Lite RAG."""

import streamlit as st

from rag import config
from rag.llm import OllamaUnavailableError
from rag.pipeline import RagPipeline

st.set_page_config(
    page_title="Assistant documentaire Lite RAG",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .answer-card {
        background: linear-gradient(135deg, #f8f9fb 0%, #eef1f6 100%);
        border: 1px solid #dde3ec;
        border-left: 5px solid #4f6df5;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .passage-card {
        background: #fafbfc;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #333;
    }
    .passage-meta {
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    model = st.text_input(
        "Modèle Ollama",
        value=config.OLLAMA_MODEL,
        help="Le modèle doit être téléchargé : `ollama pull <modele>`",
    )
    chunk_size = st.slider(
        "Taille des segments (caractères)",
        min_value=200,
        max_value=2000,
        value=config.CHUNK_SIZE,
        step=100,
        help="Modifier cette valeur reconstruit l'index vectoriel.",
    )
    top_k = st.slider(
        "Segments récupérés (top-k)",
        min_value=1,
        max_value=10,
        value=config.TOP_K,
    )

    st.divider()
    st.markdown("## 📖 À propos")
    st.markdown(
        "Pipeline **Lite RAG** : le corpus est découpé en segments, "
        "indexé dans **ChromaDB** (base vectorielle locale), puis les "
        "passages les plus proches de la question sont transmis au LLM "
        "via **Ollama** qui répond uniquement à partir de ceux-ci."
    )


@st.cache_resource(show_spinner="Indexation du corpus…")
def get_pipeline(chunk_size: int, model: str) -> RagPipeline:
    return RagPipeline(chunk_size=chunk_size, model=model)


pipeline_ready = True
try:
    pipeline = get_pipeline(int(chunk_size), model)
except Exception as exc:  # corpus manquant, etc.
    pipeline_ready = False
    st.error(f"Impossible d'initialiser le pipeline : {exc}")

with st.sidebar:
    if pipeline_ready:
        st.divider()
        st.markdown("## 🗂️ Corpus")
        st.markdown(
            f"- Fichier : `{pipeline.corpus_path}`\n"
            f"- Segments indexés : **{pipeline.n_chunks}**\n"
            f"- Taille de segment : **{pipeline.chunk_size}** caractères"
        )

# ---------------------------------------------------------------- En-tête
st.title("📚 Assistant documentaire Lite RAG")
st.markdown(
    "Posez une question en langage naturel sur le corpus fourni "
    "*(discours d'investiture présidentielle, 2013)*. L'application retrouve "
    "les passages pertinents et génère une réponse **fondée uniquement sur "
    "ceux-ci** — elle signale toute information absente du corpus."
)
st.divider()

# ---------------------------------------------------------------- Question
with st.form("question_form"):
    question = st.text_input(
        "Votre question",
        placeholder="Ex. : Que dit le discours sur le changement climatique ?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "🔍 Rechercher", type="primary", use_container_width=True
    )

if "history" not in st.session_state:
    st.session_state.history = []

if submitted and pipeline_ready:
    if not question.strip():
        st.warning("⚠️ Veuillez saisir une question avant de lancer la recherche.")
    else:
        try:
            with st.spinner("Recherche des passages et génération de la réponse…"):
                result = pipeline.ask(question.strip(), top_k=int(top_k))
        except OllamaUnavailableError as exc:
            st.error(f"🔌 {exc}")
        except Exception as exc:
            st.error(f"❌ Erreur inattendue : {exc}")
        else:
            st.session_state.history.insert(
                0,
                {
                    "Question": question.strip(),
                    "Modèle": model,
                    "Statut": result["status"],
                    "Segments": result["n_segments"],
                    "Temps (s)": result["elapsed_seconds"],
                },
            )

            if result["status"] == "hors_corpus":
                st.info("ℹ️ L'information demandée n'est pas présente dans le corpus.")

            st.markdown("### 💬 Réponse")
            st.markdown(
                f'<div class="answer-card">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ Temps de traitement", f'{result["elapsed_seconds"]} s')
            col2.metric("📄 Segments récupérés", result["n_segments"])
            col3.metric("🤖 Modèle", model)

            st.markdown("### 📎 Passages utilisés")
            for i, passage in enumerate(result["passages"], start=1):
                st.markdown(
                    f'<div class="passage-card">'
                    f'<div class="passage-meta">Passage {i} — distance : '
                    f'{passage["distance"]}</div>{passage["text"]}</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------- Historique
if st.session_state.history:
    st.divider()
    with st.expander(f"📊 Historique de la session ({len(st.session_state.history)} requête(s))"):
        st.dataframe(st.session_state.history, use_container_width=True)
        st.caption(f"Journal complet : `{config.LOG_FILE}`")
