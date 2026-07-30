"""Streamlit session_state keys and lifecycle helpers for the legal-rag-engine demo."""

from __future__ import annotations

from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000"

DOC_WIDGET_KEY = "inp_doc_pin"
MANIFEST_WIDGET_KEY = "inp_manifest_pin"


class SessionKeys:
    """Centralized keys for st.session_state."""

    base_url = "lr_base_url"
    selected_document_id = "lr_selected_document_id"
    selected_manifest_id = "lr_selected_manifest_id"
    pending_document_pin = "lr_pending_document_pin"
    pending_manifest_pin = "lr_pending_manifest_pin"
    flash_message = "lr_flash_message"
    flash_message_type = "lr_flash_message_type"
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
    ss = st.session_state
    if SessionKeys.base_url not in ss:
        ss[SessionKeys.base_url] = DEFAULT_BASE_URL
    if SessionKeys.selected_document_id not in ss:
        ss[SessionKeys.selected_document_id] = ""
    if SessionKeys.selected_manifest_id not in ss:
        ss[SessionKeys.selected_manifest_id] = ""
    if SessionKeys.health_status not in ss:
        ss[SessionKeys.health_status] = None
    if SessionKeys.health_detail not in ss:
        ss[SessionKeys.health_detail] = None


def apply_pending_pins(session_state: Any) -> None:
    """Apply queued pin values before sidebar widgets are instantiated."""
    pending_doc = session_state.pop(SessionKeys.pending_document_pin, None)
    if pending_doc is not None:
        doc_id = str(pending_doc).strip()
        session_state[SessionKeys.selected_document_id] = doc_id
        session_state[DOC_WIDGET_KEY] = doc_id

    pending_manifest = session_state.pop(SessionKeys.pending_manifest_pin, None)
    if pending_manifest is not None:
        manifest_id = str(pending_manifest).strip()
        session_state[SessionKeys.selected_manifest_id] = manifest_id
        session_state[MANIFEST_WIDGET_KEY] = manifest_id


def ensure_pin_widgets_initialized(session_state: Any) -> None:
    """Seed widget keys from canonical IDs when widgets have not run yet."""
    if DOC_WIDGET_KEY not in session_state:
        session_state[DOC_WIDGET_KEY] = session_state.get(SessionKeys.selected_document_id, "")
    if MANIFEST_WIDGET_KEY not in session_state:
        session_state[MANIFEST_WIDGET_KEY] = session_state.get(
            SessionKeys.selected_manifest_id, ""
        )


def sync_document_pin_from_widget(session_state: Any) -> None:
    """Copy manual document widget edits into the canonical selected ID."""
    session_state[SessionKeys.selected_document_id] = str(
        session_state.get(DOC_WIDGET_KEY, "")
    ).strip()


def sync_manifest_pin_from_widget(session_state: Any) -> None:
    """Copy manual manifest widget edits into the canonical selected ID."""
    session_state[SessionKeys.selected_manifest_id] = str(
        session_state.get(MANIFEST_WIDGET_KEY, "")
    ).strip()


def sync_pins_from_widgets(session_state: Any) -> None:
    """Read widget-owned keys into canonical IDs (safe after widget instantiation)."""
    sync_document_pin_from_widget(session_state)
    sync_manifest_pin_from_widget(session_state)


def queue_document_pin(session_state: Any, document_id: str) -> None:
    """Queue programmatic document pinning for the next run (no widget mutation)."""
    doc_id = str(document_id).strip()
    session_state[SessionKeys.selected_document_id] = doc_id
    session_state[SessionKeys.pending_document_pin] = doc_id


def queue_manifest_pin(session_state: Any, manifest_id: str) -> None:
    """Queue programmatic manifest pinning for the next run (no widget mutation)."""
    mid = str(manifest_id).strip()
    session_state[SessionKeys.selected_manifest_id] = mid
    session_state[SessionKeys.pending_manifest_pin] = mid


def set_flash_message(
    session_state: Any,
    message: str,
    *,
    level: str = "success",
) -> None:
    """Preserve a one-shot UI notification across ``st.rerun()``."""
    session_state[SessionKeys.flash_message] = message
    session_state[SessionKeys.flash_message_type] = level


def consume_flash_message(session_state: Any) -> tuple[str | None, str]:
    """Return and clear a queued flash message, if any."""
    message = session_state.pop(SessionKeys.flash_message, None)
    level = str(session_state.pop(SessionKeys.flash_message_type, "success"))
    return message, level
