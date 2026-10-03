"""Sidebar: the CV knowledge base. Upload CVs, process them in parallel, list, re-index and delete them."""
import streamlit as st

from cv_chat import config
from cv_chat.rag import ingest


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
            _process(files)
            cvs = _list_cvs(show_error=False)

        _show(cvs)
        if cvs:
            _manage(cvs)

        if st.button("New chat", icon=":material/add_comment:", width="stretch"):
            st.session_state.messages = []
    return cvs


def _process(files) -> None:
    """Process the uploaded files in parallel, showing each result as soon as its file finishes."""
    total = len(files)
    failed = 0
    with st.status(f"Processing {total} CVs", expanded=True) as status:
        progress = st.progress(0.0, text=f"0 of {total} done")
        try:
            for done, result in enumerate(ingest.process_cvs([(f.name, f.getvalue()) for f in files]), start=1):
                if result["error"]:
                    failed += 1
                    st.markdown(f":red[:material/error:] **{result['file']}**")
                    st.caption(result["error"].splitlines()[0][:200])
                elif result["skipped"]:
                    st.markdown(f":gray[:material/check_circle:] **{result['file']}** · already indexed, unchanged")
                else:
                    st.markdown(f":green[:material/check_circle:] **{result['file']}** · {result['chunks']} chunks")
                progress.progress(done / total, text=f"{done} of {total} done")
        except Exception as error:  # setup failed before any file ran (for example a wrong key)
            status.update(label="Processing failed", state="error")
            st.error(str(error).splitlines()[0], icon=":material/error:")
            return
        summary = f"Indexed {total - failed} of {total} CVs"
        status.update(label=summary, state="error" if failed else "complete", expanded=bool(failed))
    st.toast(summary, icon=":material/warning:" if failed else ":material/task_alt:")


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
    try:
        chunks = ingest.reindex_cv(name)
        st.session_state.notice = (f"Re-indexed {name} · {chunks} chunks", ":material/task_alt:")
    except Exception as error:
        st.session_state.notice = (f"Could not re-index {name}: {str(error).splitlines()[0][:150]}", ":material/error:")


def _delete(name: str) -> None:
    try:
        ingest.delete_cv(name)
        st.session_state.notice = (f"Deleted {name}", ":material/task_alt:")
    except Exception as error:
        st.session_state.notice = (f"Could not delete {name}: {str(error).splitlines()[0][:150]}", ":material/error:")
