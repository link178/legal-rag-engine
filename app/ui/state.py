"""Streamlit session_state keys and defaults for the legal-rag-engine demo."""

from __future__ import annotations

from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"


class SessionKeys:
    """Centralized keys for st.session_state."""

    base_url = "lr_base_url"
    selected_document_id = "lr_selected_document_id"
    selected_manifest_id = "lr_selected_manifest_id"
    health_status = "lr_health_status"
    health_detail = "lr_health_detail"
    last_ingest = "lr_last_ingest"
    last_chunk = "lr_last_chunk"
    last_index = "lr_last_index"
    last_manifests = "lr_last_manifests"
    last_retrieve = "lr_last_retrieve"
    last_answer = "lr_last_answer"


def init_state(st: Any) -> None:
    """Initialize session_state once (call with streamlit module as st)."""
    if SessionKeys.base_url not in st.session_state:
        st.session_state[SessionKeys.base_url] = DEFAULT_BASE_URL
    if SessionKeys.selected_document_id not in st.session_state:
        st.session_state[SessionKeys.selected_document_id] = ""
    if SessionKeys.selected_manifest_id not in st.session_state:
        st.session_state[SessionKeys.selected_manifest_id] = ""
    if SessionKeys.health_status not in st.session_state:
        st.session_state[SessionKeys.health_status] = None
    if SessionKeys.health_detail not in st.session_state:
        st.session_state[SessionKeys.health_detail] = None
