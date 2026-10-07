"""The app's stylesheet, added to every page (login included) from app.py."""
from pathlib import Path

import streamlit as st

_CSS = Path(__file__).with_name("styles.css")
READING_WIDTH = "52rem"  # the chat is a comfortable reading column; the Candidates grid uses the full width


def inject(reading_column: bool = False) -> None:
    st.html(_CSS)
    if reading_column:
        st.html(f"<style>[data-testid='stMainBlockContainer'] {{ max-width: {READING_WIDTH}; }}</style>")
