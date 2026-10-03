"""Sidebar: the CV knowledge base. Upload CVs, watch them being processed in parallel, list, re-index and delete them."""
import streamlit as st

from cv_chat import config
from cv_chat.rag import ingest, jobs


def render() -> list[str]:
    """Draw the sidebar and return the names of the indexed CVs."""
    if notice := st.session_state.pop("notice", None):  # result of a Manage action, which ran before this rerun
        st.toast(notice[0], icon=notice[1])
    with st.sidebar:
        st.markdown("### :material/folder_open: Knowledge base")
        cvs = _list_cvs()
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
            _start([(f.name, f.getvalue()) for f in files])
        _status_panel()

        _show(cvs)
        if cvs:
            _manage(cvs)

        if st.button("New chat", icon=":material/add_comment:", width="stretch"):
            st.session_state.messages = []
    return cvs


def _start(files: list[tuple[str, bytes]]) -> None:
    """Queue the files. They are processed in the background, several at a time, and show up in the status panel."""
    try:
        jobs.queue.submit(files)
    except Exception as error:  # setup failed before any file ran (for example a wrong key)
        st.error(str(error).splitlines()[0], icon=":material/error:")


@st.fragment(run_every=2)
def _status_panel() -> None:
    """Live state of every file, refreshed every 2 seconds. When the last file finishes, the whole app reruns once."""
    snapshot = jobs.queue.snapshot()
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
            st.button("Clear status", icon=":material/close:", on_click=jobs.queue.clear_finished, width="stretch")

    active = jobs.queue.active()
    if st.session_state.get("ingest_was_active") and not active:
        failed = sum(job.state == "failed" for job in snapshot)
        st.session_state.notice = (
            (f"{failed} of {len(snapshot)} CVs failed", ":material/warning:")
            if failed
            else (f"Processed {len(snapshot)} CVs", ":material/task_alt:")
        )
        st.session_state.ingest_was_active = False
        st.rerun()  # refresh the list of indexed CVs and enable the chat
    st.session_state.ingest_was_active = active


def _show_job(job: jobs.Job) -> None:
    if job.state == "queued":
        st.markdown(f":gray[:material/schedule:] **{job.file_name}** · waiting")
    elif job.state == "processing":
        st.markdown(f":blue[:material/sync:] **{job.file_name}** · {job.stage}")
    elif job.state == "indexed":
        st.markdown(f":green[:material/check_circle:] **{job.file_name}** · {job.chunks} chunks")
    elif job.state == "skipped":
        st.markdown(f":gray[:material/check_circle:] **{job.file_name}** · already indexed, unchanged")
    else:
        st.markdown(f":red[:material/error:] **{job.file_name}**")
        st.caption((job.error.splitlines() or ["Failed"])[0][:200])


def _list_cvs(show_error: bool = True) -> list[str]:
    try:
        return ingest.list_cvs()
    except Exception as error:
        if show_error:
            st.error(
                f"Could not list the CVs: {str(error).splitlines()[0]}. Check AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER in .env.",
                icon=":material/error:",
            )
        return []


def _show(cvs: list[str]) -> None:
    st.badge(f"{len(cvs)} CV{'' if len(cvs) == 1 else 's'} indexed", icon=":material/database:", color="primary")
    if cvs:
        with st.container(border=True, height=260 if len(cvs) > 7 else "content"):
            for name in cvs:
                st.markdown(f":material/description: {name}")


def _manage(cvs: list[str]) -> None:
    """Re-index or delete one CV. The buttons act in callbacks, so the list above is drawn after the change."""
    with st.expander("Manage a CV", icon=":material/tune:"):
        target = st.selectbox("CV", cvs, label_visibility="collapsed")
        reindex_column, delete_column = st.columns(2)
        reindex_column.button(
            "Re-index", icon=":material/refresh:", width="stretch", on_click=_reindex, args=(target,),
            help="Process the stored file again, for example after the pipeline changed",
        )
        with delete_column.popover("Delete", icon=":material/delete:", width="stretch"):
            st.write(f"Delete **{target}** from storage and from search?")
            st.button("Yes, delete", type="primary", icon=":material/delete:", on_click=_delete, args=(target,))


def _reindex(name: str) -> None:
    """Queue the stored original for processing again; the status panel shows its progress."""
    try:
        jobs.queue.submit([(name, None)], force=True)
    except Exception as error:
        st.session_state.notice = (f"Could not re-index {name}: {str(error).splitlines()[0][:150]}", ":material/error:")


def _delete(name: str) -> None:
    try:
        ingest.delete_cv(name)
        jobs.queue.forget(name)
        st.session_state.notice = (f"Deleted {name}", ":material/task_alt:")
    except Exception as error:
        st.session_state.notice = (f"Could not delete {name}: {str(error).splitlines()[0][:150]}", ":material/error:")
