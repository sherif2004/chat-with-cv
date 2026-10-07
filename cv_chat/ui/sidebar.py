"""Sidebar: the CV knowledge base. Upload CVs, watch them being processed in parallel, list, re-index and delete them."""
import time

import streamlit as st

from cv_chat import config
from cv_chat.rag import ingest, jobs
from cv_chat.rag.cache import cache_for
from cv_chat.ui import chats
from cv_chat.ui.safe import esc
from cv_chat.workspace import Workspace


def render(ws: Workspace) -> list[str]:
    """Draw the sidebar and return the names of the indexed CVs."""
    if notice := st.session_state.pop("notice", None):  # result of a Manage action, which ran before this rerun
        st.toast(esc(notice[0]), icon=notice[1])
    with st.sidebar:
        chats.render(ws)
        st.markdown("### :material/folder_open: Knowledge base")
        cvs = _list_cvs(ws)
        st.caption(f"Upload at least {config.MIN_CVS} CVs")
        files = st.file_uploader(
            "CV files", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed"
        )
        new_names = {f.name for f in files} - set(cvs)  # uploading an indexed file again adds nothing
        total = len(cvs) + len(new_names)
        enough = total >= config.MIN_CVS
        if files and not enough:
            st.warning(f"{total} of {config.MIN_CVS} CVs. Add {config.MIN_CVS - total} more to continue.", icon=":material/info:")
        if st.button("Process CVs", icon=":material/bolt:", type="primary", disabled=not files or not enough, width="stretch"):
            _start(ws, [(f.name, f.getvalue()) for f in files])
        _status_panel(ws)

        _show(cvs)
        if cvs:
            _manage(ws, cvs)

        if cvs:
            _scope(cvs)

        st.toggle(
            "Query expansion", key="expand_queries",
            help="Also search two reworded versions of each question and merge the results. "
            "Finds more, but answers start 1 to 2 seconds later and each question uses three semantic searches.",
        )
        st.toggle(
            "Cache final answers", key="cache_answers",
            help="Reuse the answer to an identical question in an identical chat. Off by default: a cached answer can be out of date.",
        )
        st.button(
            "Clear cache", icon=":material/mop:", width="stretch", on_click=_clear_cache, args=(ws,),
            help="Forget cached router results, searches and answers. This also happens whenever a CV is processed or deleted.",
        )
    return cvs


def _clear_cache(ws: Workspace) -> None:
    cache_for(ws).clear()
    st.session_state.notice = ("Cache cleared", ":material/task_alt:")


def _start(ws: Workspace, files: list[tuple[str, bytes]]) -> None:
    """Queue the files. They are processed in the background, several at a time, and show up in the status panel."""
    try:
        jobs.queue_for(ws).submit(files)
    except Exception as error:  # setup failed before any file ran (for example a wrong key)
        st.error(str(error).splitlines()[0], icon=":material/error:")


@st.fragment(run_every=2)
def _status_panel(ws: Workspace) -> None:
    """Live state of every file, refreshed every 2 seconds. When the last file finishes, the whole app reruns once."""
    queue = jobs.queue_for(ws)
    snapshot = queue.snapshot()
    if snapshot:
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
            st.button("Clear status", icon=":material/close:", on_click=queue.clear_finished, width="stretch")

    active = queue.active()
    if st.session_state.get("ingest_was_active") and not active:
        failed = sum(job.state == "failed" for job in snapshot)
        st.session_state.notice = (
            (f"{failed} of {len(snapshot)} CVs failed", ":material/warning:")
            if failed
            else (f"Processed {len(snapshot)} CVs", ":material/task_alt:")
        )
        st.session_state.ingest_was_active = False
        _forget_cv_list()
        st.rerun()  # refresh the list of indexed CVs and enable the chat
    st.session_state.ingest_was_active = active


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


CV_LIST_SECONDS = 300  # how long the list of CVs is kept before Azure is asked again (it is also dropped whenever it changes)


def _forget_cv_list() -> None:
    st.session_state.pop("cv_list", None)


def _list_cvs(ws: Workspace, show_error: bool = True) -> list[str]:
    """The CV names. Asking Azure Blob Storage on every click made the app slow, so the list is kept for a few minutes."""
    cached = st.session_state.get("cv_list")
    if cached and time.monotonic() - cached[0] < CV_LIST_SECONDS:
        return cached[1]
    try:
        names = ingest.list_cvs(ws)
        st.session_state.cv_list = (time.monotonic(), names)
        return names
    except Exception as error:
        if show_error:
            st.error(
                f"Could not list the CVs: {str(error).splitlines()[0]}. Check AZURE_STORAGE_CONNECTION_STRING in .env.",
                icon=":material/error:",
            )
        return []


def _show(cvs: list[str]) -> None:
    st.badge(f"{len(cvs)} CV{'' if len(cvs) == 1 else 's'} indexed", icon=":material/database:", color="primary")
    if cvs:
        with st.container(border=True, height=260 if len(cvs) > 7 else "content"):
            for name in cvs:
                st.markdown(f":material/description: {esc(name)}")


def _scope(cvs: list[str]) -> None:
    """Choose which CVs the chat answers from. Nothing selected means all of them."""
    st.session_state.chat_scope = [name for name in st.session_state.get("chat_scope", []) if name in cvs]  # drop deleted CVs
    st.multiselect(
        "Chat with", cvs, key="chat_scope", placeholder="All CVs",
        help="Pick one or more CVs to answer only from them. Leave empty to search all CVs.",
    )


def _manage(ws: Workspace, cvs: list[str]) -> None:
    """Re-index or delete one CV. The buttons act in callbacks, so the list above is drawn after the change."""
    with st.expander("Manage a CV", icon=":material/tune:"):
        target = st.selectbox("CV", cvs, label_visibility="collapsed")
        reindex_column, delete_column = st.columns(2)
        reindex_column.button(
            "Re-index", icon=":material/refresh:", width="stretch", on_click=_reindex, args=(ws, target),
            help="Process the stored file again, for example after the pipeline changed",
        )
        with delete_column.popover("Delete", icon=":material/delete:", width="stretch"):
            st.write(f"Delete **{esc(target)}** from storage and from search?")
            st.button("Yes, delete", type="primary", icon=":material/delete:", on_click=_delete, args=(ws, target))
        st.button(
            "Update outdated CVs", icon=":material/published_with_changes:", width="stretch", on_click=_update_all, args=(ws, cvs),
            help="Check every CV against the current pipeline and re-index only those processed by an older version. "
            "The status panel shows the unchanged ones as skipped.",
        )


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
        _forget_cv_list()
        jobs.queue_for(ws).forget(name)
        st.session_state.notice = (f"Deleted {name}", ":material/task_alt:")
    except Exception as error:
        st.session_state.notice = (f"Could not delete {name}: {str(error).splitlines()[0][:150]}", ":material/error:")
