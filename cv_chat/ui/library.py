"""Library view: upload CVs, watch them being processed in parallel, and open, re-index or delete a CV."""
import time

import pandas as pd
import streamlit as st

from cv_chat import config
from cv_chat.rag import ingest, jobs
from cv_chat.services import blob_storage
from cv_chat.ui import candidates
from cv_chat.ui.safe import esc
from cv_chat.workspace import Workspace

CV_LIST_SECONDS = 300  # how long the list of CVs is kept before Azure is asked again (it is also dropped whenever it changes)


def forget_cv_list() -> None:
    st.session_state.pop("cv_list", None)


def cv_names(ws: Workspace) -> list[str]:
    """The CV names. Asking Azure Blob Storage on every click made the app slow, so the list is kept for a few minutes."""
    cached = st.session_state.get("cv_list")
    if cached and time.monotonic() - cached[0] < CV_LIST_SECONDS:
        return cached[1]
    try:
        names = ingest.list_cvs(ws)
        st.session_state.cv_list = (time.monotonic(), names)
        return names
    except Exception as error:
        st.error(
            f"Could not list the CVs: {str(error).splitlines()[0]}. Check AZURE_STORAGE_CONNECTION_STRING in .env.",
            icon=":material/error:",
        )
        return []


def render(ws: Workspace, cvs: list[str]) -> None:
    st.markdown("#### Library")
    st.caption(f"{len(cvs)} CV{'' if len(cvs) == 1 else 's'} indexed · at least {config.MIN_CVS} are needed to chat")

    with st.container(border=True):
        files = st.file_uploader("CV files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="library_files")
        new_names = {f.name for f in files} - set(cvs)  # uploading an indexed file again adds nothing
        total = len(cvs) + len(new_names)
        enough = total >= config.MIN_CVS
        if files and not enough:
            st.warning(f"{total} of {config.MIN_CVS} CVs. Add {config.MIN_CVS - total} more to continue.", icon=":material/info:")
        st.button("Process CVs", icon=":material/bolt:", type="primary", disabled=not files or not enough, on_click=_start, args=(ws,))
    progress(ws)

    if not cvs:
        st.info("No CVs yet. Add some above to get started.", icon=":material/upload_file:")
        return
    _table(ws, cvs)


@st.fragment(run_every=2)
def progress(ws: Workspace) -> None:
    """The live state of every file, refreshed every 2 seconds."""
    queue = jobs.queue_for(ws)
    snapshot = queue.snapshot()
    if not snapshot:
        return
    finished = [job for job in snapshot if job.state in jobs.FINISHED]
    running = sum(job.state == "processing" for job in snapshot)
    waiting = sum(job.state == "queued" for job in snapshot)
    st.progress(len(finished) / len(snapshot), text=f"{len(finished)} of {len(snapshot)} done")
    if len(finished) < len(snapshot):
        st.caption(f"{running} running in parallel · {waiting} waiting")
    with st.container(border=True):
        for job in snapshot:
            _show_job(job)
    if len(finished) == len(snapshot):
        st.button("Clear status", icon=":material/close:", on_click=queue.clear_finished)


def _show_job(job: jobs.Job) -> None:
    if job.state == "queued":
        st.markdown(f":gray[:material/schedule:] **{esc(job.file_name)}** · waiting")
    elif job.state == "processing":
        st.markdown(f":blue[:material/sync:] **{esc(job.file_name)}** · {esc(job.stage)}")
    elif job.state == "indexed":
        st.markdown(f":green[:material/check_circle:] **{esc(job.file_name)}** · {job.chunks} chunks")
    elif job.state == "skipped":
        st.markdown(f":gray[:material/check_circle:] **{esc(job.file_name)}** · already indexed, unchanged")
    else:
        st.markdown(f":red[:material/error:] **{esc(job.file_name)}**")
        st.caption(esc((job.error.splitlines() or ["Failed"])[0][:200]))


@st.fragment(run_every=2)
def watch(ws: Workspace) -> None:
    """Runs in the sidebar on every page, checking every 2 seconds. It shows a progress bar while CVs are processed, and when
    the last one finishes it refreshes the CV list and tells the user, whichever page they are on."""
    queue = jobs.queue_for(ws)
    snapshot = queue.snapshot()
    active = queue.active()
    if active:
        finished = sum(job.state in jobs.FINISHED for job in snapshot)
        st.progress(finished / len(snapshot), text=f"Processing CVs · {finished} of {len(snapshot)}")
    if st.session_state.get("ingest_was_active") and not active:
        failed = sum(job.state == "failed" for job in snapshot)
        st.session_state.notice = (
            (f"{failed} of {len(snapshot)} CVs failed", ":material/warning:")
            if failed
            else (f"Processed {len(snapshot)} CVs", ":material/task_alt:")
        )
        st.session_state.ingest_was_active = False
        forget_cv_list()
        st.rerun()  # refresh the list of indexed CVs and enable the chat
    st.session_state.ingest_was_active = active


def _start(ws: Workspace) -> None:
    """Queue the uploaded files. It runs as the button's callback, before the page is drawn, so the sidebar already sees the
    new jobs and keeps watching them. The files are processed in the background, several at a time."""
    files = [(f.name, f.getvalue()) for f in st.session_state.get("library_files", [])]
    try:
        jobs.queue_for(ws).submit(files)
    except Exception as error:  # setup failed before any file ran (for example a wrong key)
        st.session_state.notice = (str(error).splitlines()[0][:200], ":material/error:")


def _table(ws: Workspace, cvs: list[str]) -> None:
    """Every CV in a table; picking a row shows what can be done with that CV."""
    rows = candidates.candidate_list(ws, cvs)
    frame = pd.DataFrame({
        "File": [r["file_name"] for r in rows],
        "Candidate": [r["name"] for r in rows],
        "Job title": [r["title"] for r in rows],
        "Years": [r["years"] for r in rows],
    })
    top_left, top_right = st.columns([3, 2], vertical_alignment="center")
    top_left.caption("Select a CV to open, re-index or delete it.")
    top_right.button(
        "Update outdated CVs", icon=":material/published_with_changes:", width="stretch", on_click=_update_all, args=(ws, cvs),
        help="Check every CV against the current pipeline and re-index only those processed by an older version. "
        "The status panel shows the unchanged ones as skipped.",
    )
    event = st.dataframe(
        frame, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="library_table",
        column_config={"Years": st.column_config.NumberColumn(format="%g")},
    )
    chosen = event.selection.rows
    if not chosen or chosen[0] >= len(rows):
        return
    name = rows[chosen[0]]["file_name"]
    with st.container(border=True):
        st.markdown(f":material/description: **{esc(name)}**")
        open_column, reindex_column, delete_column = st.columns(3)
        if link := _link(ws, name):
            open_column.link_button("Open CV", link, icon=":material/open_in_new:", width="stretch")
        reindex_column.button(
            "Re-index", icon=":material/refresh:", width="stretch", on_click=_reindex, args=(ws, name),
            help="Process the stored file again, for example after the pipeline changed",
        )
        with delete_column.popover("Delete", icon=":material/delete:", width="stretch"):
            st.write(f"Delete **{esc(name)}** from storage and from search? This cannot be undone.")
            st.button("Yes, delete", type="primary", icon=":material/delete:", on_click=_delete, args=(ws, name))


def _link(ws: Workspace, file_name: str) -> str | None:
    try:
        return blob_storage.read_link(ws, file_name)
    except Exception:  # a missing link must never break the page
        return None


def _update_all(ws: Workspace, names: list[str]) -> None:
    """Queue every stored CV without forcing: those the current pipeline already indexed are skipped, the rest are re-indexed."""
    try:
        jobs.queue_for(ws).submit([(name, None) for name in names], force=False)
    except Exception as error:
        st.session_state.notice = (f"Could not update the CVs: {str(error).splitlines()[0][:150]}", ":material/error:")


def _reindex(ws: Workspace, name: str) -> None:
    """Queue the stored original for processing again; the status panel shows its progress."""
    try:
        jobs.queue_for(ws).submit([(name, None)], force=True)
    except Exception as error:
        st.session_state.notice = (f"Could not re-index {name}: {str(error).splitlines()[0][:150]}", ":material/error:")


def _delete(ws: Workspace, name: str) -> None:
    try:
        ingest.delete_cv(ws, name)
        forget_cv_list()
        jobs.queue_for(ws).forget(name)
        st.session_state.notice = (f"Deleted {name}", ":material/task_alt:")
    except Exception as error:
        st.session_state.notice = (f"Could not delete {name}: {str(error).splitlines()[0][:150]}", ":material/error:")
