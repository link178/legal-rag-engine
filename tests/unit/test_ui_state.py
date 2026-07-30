"""Regression tests for Streamlit pin widget lifecycle helpers."""

from __future__ import annotations

from app.ui.state import (
    DOC_WIDGET_KEY,
    MANIFEST_WIDGET_KEY,
    SessionKeys,
    apply_pending_pins,
    ensure_pin_widgets_initialized,
    queue_document_pin,
    queue_manifest_pin,
    sync_document_pin_from_widget,
    sync_pins_from_widgets,
)


def _empty_state() -> dict[str, object]:
    return {
        SessionKeys.selected_document_id: "",
        SessionKeys.selected_manifest_id: "",
    }


def test_pending_document_pin_applied_before_widget_creation() -> None:
    ss = _empty_state()
    ss[SessionKeys.pending_document_pin] = "doc-123"

    apply_pending_pins(ss)

    assert ss[SessionKeys.selected_document_id] == "doc-123"
    assert ss[DOC_WIDGET_KEY] == "doc-123"
    assert SessionKeys.pending_document_pin not in ss


def test_pending_manifest_pin_applied_before_widget_creation() -> None:
    ss = _empty_state()
    ss[SessionKeys.pending_manifest_pin] = "manifest-456"

    apply_pending_pins(ss)

    assert ss[SessionKeys.selected_manifest_id] == "manifest-456"
    assert ss[MANIFEST_WIDGET_KEY] == "manifest-456"
    assert SessionKeys.pending_manifest_pin not in ss


def test_pending_state_is_consumed_after_application() -> None:
    ss = _empty_state()
    queue_document_pin(ss, "doc-a")
    queue_manifest_pin(ss, "manifest-b")

    apply_pending_pins(ss)

    assert SessionKeys.pending_document_pin not in ss
    assert SessionKeys.pending_manifest_pin not in ss

    apply_pending_pins(ss)

    assert ss[SessionKeys.selected_document_id] == "doc-a"
    assert ss[SessionKeys.selected_manifest_id] == "manifest-b"


def test_manual_widget_edits_sync_to_canonical_ids() -> None:
    ss = _empty_state()
    ss[DOC_WIDGET_KEY] = "manual-doc"
    ss[MANIFEST_WIDGET_KEY] = "manual-manifest"

    sync_pins_from_widgets(ss)

    assert ss[SessionKeys.selected_document_id] == "manual-doc"
    assert ss[SessionKeys.selected_manifest_id] == "manual-manifest"


def test_manual_document_on_change_sync() -> None:
    ss = _empty_state()
    ss[DOC_WIDGET_KEY] = "typed-doc"

    sync_document_pin_from_widget(ss)

    assert ss[SessionKeys.selected_document_id] == "typed-doc"


def test_programmatic_pinning_does_not_touch_widget_keys() -> None:
    ss = _empty_state()
    ss[DOC_WIDGET_KEY] = "existing-doc-widget"
    ss[MANIFEST_WIDGET_KEY] = "existing-manifest-widget"

    queue_document_pin(ss, "new-doc")
    queue_manifest_pin(ss, "new-manifest")

    assert ss[DOC_WIDGET_KEY] == "existing-doc-widget"
    assert ss[MANIFEST_WIDGET_KEY] == "existing-manifest-widget"
    assert ss[SessionKeys.selected_document_id] == "new-doc"
    assert ss[SessionKeys.selected_manifest_id] == "new-manifest"
    assert ss[SessionKeys.pending_document_pin] == "new-doc"
    assert ss[SessionKeys.pending_manifest_pin] == "new-manifest"


def test_document_and_manifest_flows_share_semantics() -> None:
    ss = _empty_state()

    queue_document_pin(ss, "doc-x")
    apply_pending_pins(ss)
    ensure_pin_widgets_initialized(ss)

    queue_manifest_pin(ss, "manifest-y")
    apply_pending_pins(ss)
    ensure_pin_widgets_initialized(ss)

    assert ss[SessionKeys.selected_document_id] == "doc-x"
    assert ss[SessionKeys.selected_manifest_id] == "manifest-y"
    assert ss[DOC_WIDGET_KEY] == "doc-x"
    assert ss[MANIFEST_WIDGET_KEY] == "manifest-y"


def test_rerun_does_not_repeat_api_when_only_pending_pin_is_queued() -> None:
    """Simulate ingest success: store result, queue pin, rerun; no second API call."""
    ss = _empty_state()
    api_calls = {"count": 0}

    def ingest_once() -> dict[str, str]:
        api_calls["count"] += 1
        return {"document_id": "doc-ingested"}

    data = ingest_once()
    ss[SessionKeys.last_ingest] = data
    queue_document_pin(ss, str(data["document_id"]))

    apply_pending_pins(ss)
    ensure_pin_widgets_initialized(ss)

    assert api_calls["count"] == 1
    assert ss[SessionKeys.last_ingest] == {"document_id": "doc-ingested"}
    assert ss[DOC_WIDGET_KEY] == "doc-ingested"


def test_ensure_pin_widgets_initialized_seeds_from_canonical() -> None:
    ss = _empty_state()
    ss[SessionKeys.selected_document_id] = "canonical-doc"
    ss[SessionKeys.selected_manifest_id] = "canonical-manifest"

    ensure_pin_widgets_initialized(ss)

    assert ss[DOC_WIDGET_KEY] == "canonical-doc"
    assert ss[MANIFEST_WIDGET_KEY] == "canonical-manifest"
