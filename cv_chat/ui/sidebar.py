"""Sidebar: the CV knowledge base. Upload CVs, process them in parallel, and list what is indexed."""
import streamlit as st

from cv_chat.rag import ingest


def render() -> list[str]:
    """Draw the sidebar and return the names of the indexed CVs."""
    with st.sidebar:
        st.markdown("### :material/folder_open: Knowledge base")
        st.caption("Upload at least 8 CVs")
        files = st.file_uploader(
            "CV files", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed"
        )
        if st.button("Process CVs", icon=":material/bolt:", type="primary", disabled=not files, width="stretch"):
            _process(files)

        cvs = _indexed_cvs()

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


def _indexed_cvs() -> list[str]:
    try:
        cvs = ingest.list_cvs()
    except Exception as error:
        st.error(f"Could not list the CVs: {str(error).splitlines()[0]}", icon=":material/error:")
        return []
    st.badge(f"{len(cvs)} CV{'' if len(cvs) == 1 else 's'} indexed", icon=":material/database:", color="primary")
    if cvs:
        with st.container(border=True, height=260 if len(cvs) > 7 else "content"):
            for name in cvs:
                st.markdown(f":material/description: {name}")
    return cvs
