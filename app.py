"""Interface utilisateur Streamlit.

Lancement :  streamlit run app.py

L'interface est volontairement minimale :
  - colonne de gauche : gestion des documents (import, indexation, suppression) ;
  - zone principale : poser une question et lire la réponse avec ses sources.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from rag import RAGPipeline, Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="RAG Mistral + Qdrant", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------
# Initialisation (mise en cache : le pipeline n'est construit qu'une fois
# pour toute la session, pas à chaque interaction).
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialisation du pipeline...")
def get_pipeline() -> tuple[RAGPipeline, Settings]:
    settings = Settings.from_env()
    return RAGPipeline.from_settings(settings), settings


try:
    pipeline, settings = get_pipeline()
except Exception as error:  # noqa: BLE001
    st.error(f"**Impossible de démarrer l'application**\n\n{error}")
    st.info(
        "Vérifiez que :\n"
        "1. Qdrant tourne (`docker compose up -d`)\n"
        "2. Le fichier `.env` contient une clé `MISTRAL_API_KEY` valide"
    )
    st.stop()

data_dir = Path(settings.data_dir)
data_dir.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Barre latérale : gestion des documents
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Importer des PDF", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("Indexer", type="primary", use_container_width=True):
        progress = st.progress(0.0)
        for position, uploaded in enumerate(uploaded_files, start=1):
            destination = data_dir / uploaded.name
            destination.write_bytes(uploaded.getbuffer())
            try:
                nb_chunks = pipeline.ingest_file(destination)
                st.success(f"{uploaded.name} : {nb_chunks} chunk(s)")
            except Exception as error:  # noqa: BLE001
                st.error(f"{uploaded.name} : {error}")
            progress.progress(position / len(uploaded_files))
        st.rerun()

#-----------
    monbouton = st.button("Cliquezzz")
    with monbouton:
        from pathlib import Path
        chemin = Path("./EnterpriseRAG-bench")
        nombre_files = len(list(chemin.rglob("*.txt")))
        progress = st.progress(0.0)
        i=0
        for fichier in chemin.rglob("*.txt"):
            destination = data_dir / fichier.name
            destination.write_bytes(fichier.getbuffer())
            try:
                nb_chunks = pipeline.ingest_file(destination)
                st.success(f"{fichier.name} : {nb_chunks} chunk(s)")
            except Exception as error:  # noqa: BLE001
                st.error(f"{uploaded.name} : {error}")
            i+=1
        progress.progress(i / nombre_files)

#-----------

    st.divider()

    # Inventaire de ce qui est réellement présent dans la base vectorielle.
    try:
        sources = pipeline.vector_store.list_sources()
        total_chunks = pipeline.vector_store.count()
    except Exception as error:  # noqa: BLE001
        sources, total_chunks = [], 0
        st.warning(f"Lecture de la base impossible : {error}")

    st.caption(f"**{len(sources)}** document(s) · **{total_chunks}** chunk(s) indexés")

    for source in sources:
        col_name, col_delete = st.columns([4, 1])
        col_name.write(f"• {source}")
        if col_delete.button("🗑", key=f"del-{source}", help=f"Supprimer {source}"):
            pipeline.vector_store.delete_source(source)
            st.rerun()

    if sources:
        st.divider()
        if st.button("Vider la base", use_container_width=True):
            pipeline.vector_store.reset()
            st.rerun()

    st.divider()
    st.caption(
        f"Embeddings : `{settings.embedding_model}`  \n"
        f"LLM : `{settings.llm_model}`  \n"
        f"Chunks récupérés : `{settings.top_k}`"
    )


# ----------------------------------------------------------------------
# Zone principale : question / réponse
# ----------------------------------------------------------------------
st.title("📚 Assistant documentaire")
st.caption("Posez une question ; la réponse est construite à partir de vos PDF uniquement.")

question = st.text_input(
    "Votre question",
    placeholder="Ex. : Quelles sont les conclusions du rapport ?",
)

if st.button("Rechercher", type="primary", disabled=not question):
    if not sources:
        st.warning("Aucun document indexé. Importez d'abord un PDF dans la colonne de gauche.")
    else:
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                answer = pipeline.ask(question)
            except Exception as error:  # noqa: BLE001
                st.error(f"Erreur : {error}")
                answer = None

        if answer:
            st.subheader("Réponse")
            st.write(answer.text)

            st.subheader("Sources")
            if not answer.sources:
                st.info("Aucun passage pertinent trouvé.")

            for position, result in enumerate(answer.sources, start=1):
                chunk = result.chunk
                titre = f"[{position}] {chunk.reference} — similarité {result.score:.3f}"
                with st.expander(titre):
                    st.markdown(f"> {chunk.text}")
