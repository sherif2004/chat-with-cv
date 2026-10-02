import logging
from dataclasses import astuple

import pandas as pd
import streamlit as st

from src.application.ingest_cv import Deps, IngestionService
from src.config import load_settings, require_ingest, setup_logging
from src.domain.models import CVInfo
from src.infrastructure.blob_store import AzureBlobStore
from src.infrastructure.docling_extractor import DoclingExtractor
from src.infrastructure.openai_client import AzureEmbedder
from src.infrastructure.search_index import AzureSearchIndex

st.set_page_config(page_title="CV Ingestion", page_icon="📄", layout="wide")
setup_logging()
log = logging.getLogger("src.app")
MIN_CVS = 8


@st.cache_resource(show_spinner="Loading layout models (first run downloads them)…")
def get_services() -> tuple[Deps, IngestionService]:
    settings = load_settings()
    require_ingest(settings)
    deps = Deps(
        blob=AzureBlobStore(settings),
        extractor=DoclingExtractor(settings.docling_device),
        embedder=AzureEmbedder(settings),
        index=AzureSearchIndex(settings),
    )
    deps.extractor.warm_up()
    deps.index.ensure()
    return deps, IngestionService(deps, settings.ingest_workers)


try:
    deps, service = get_services()
except Exception as e:
    st.error(f"Setup problem: {e}")
    st.stop()


@st.cache_data(ttl=30, show_spinner=False)
def indexed_cvs(_index, version: int):
    """Cached per ingest/delete `version`; the short TTL covers search-index propagation delay."""
    return [astuple(c) for c in _index.list_cvs()]  # plain tuples: dataclasses can't be pickled across Streamlit reloads


@st.fragment(run_every=2)
def status_panel() -> None:
    """Polls job state; reruns the whole app once when the last job finishes."""
    jobs = service.snapshot()
    if jobs:
        st.dataframe(
            pd.DataFrame([{"file": j.filename, "status": j.state, "chunks": j.chunks, "error": j.error} for j in jobs]),
            hide_index=True,
            use_container_width=True,
        )
    active = service.active()
    if st.session_state.get("was_active") and not active:
        st.session_state.was_active = False
        st.rerun()
    st.session_state.was_active = active


st.title("CV ingestion")
st.caption("Upload PDFs → Blob Storage → extract → chunk → embed → Azure AI Search")

try:
    cvs = [CVInfo(*row) for row in indexed_cvs(deps.index, service.version)]
except Exception as e:
    cvs = []
    st.warning(f"Could not list indexed CVs: {e}")

uploads = st.file_uploader("Upload CVs (PDF)", type="pdf", accept_multiple_files=True)
new_names = {f.name for f in uploads} - {c.filename for c in cvs}
total = len(cvs) + len(new_names)  # a re-upload of an indexed file adds nothing
ready = bool(uploads) and total >= MIN_CVS
if uploads and not ready:
    st.warning(f"Need at least {MIN_CVS} CVs in total ({len(cvs)} indexed + {len(new_names)} new selected).")
if st.button("Index uploaded CVs", disabled=not ready, type="primary"):
    service.submit([(f.name, f.getvalue()) for f in uploads])
status_panel()

st.subheader(f"Indexed CVs ({len(cvs)})")
if cvs:
    st.dataframe(
        pd.DataFrame([{"file": c.filename, "chunks": c.chunks} for c in cvs]),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Manage"):
        target = st.selectbox("CV", [c.filename for c in cvs])
        left, right = st.columns(2)
        if left.button("Re-index"):
            service.reindex(target)
            st.rerun()
        if right.button("Delete"):
            service.delete(target)
            st.rerun()
else:
    st.info("Nothing indexed yet.")
