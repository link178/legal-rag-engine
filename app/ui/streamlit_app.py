"""legal-rag-engine: minimal Streamlit demo over the FastAPI v1 HTTP API."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from app.ui.api_client import ApiError, LegalRagApiClient
from app.ui.state import SessionKeys, init_state

RETRIEVAL_MODES: tuple[str, ...] = ("hybrid", "dense_only", "sparse_only")
CHUNK_STRATEGIES: tuple[str, ...] = ("fixed_size", "structure_aware")

_METADATA_FILTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("corpus_name", "corpus_name"),
    ("corpus_adapter", "corpus_adapter"),
    ("source_family", "source_family"),
    ("jurisdiction", "jurisdiction"),
    ("legal_document_type", "legal_document_type"),
    ("language", "language"),
    ("canonical_id", "canonical_id"),
)


def _metadata_filter_from_expander(key_prefix: str) -> dict[str, str] | None:
    """Build API ``metadata_filter`` from non-empty Streamlit fields (Phase 14)."""
    out: dict[str, str] = {}
    with st.expander("Metadata filter (optional)"):
        for json_key, label in _METADATA_FILTER_FIELDS:
            v = st.text_input(
                label,
                value="",
                key=f"{key_prefix}_mf_{json_key}",
            )
            if (v or "").strip():
                out[json_key] = v.strip()
    return out or None


def _client_for_base(url: str) -> LegalRagApiClient:
    return LegalRagApiClient(url.rstrip("/") or "http://localhost:8000")


def render_api_error(err: ApiError) -> None:
    """Map API error codes to Streamlit feedback."""
    code = err.code
    msg = err.message
    detail_bits = []
    if err.details:
        detail_bits.append(json.dumps(err.details, indent=2)[:4000])
    body = f"**{code}** — {msg}"
    if detail_bits:
        body += f"\n\n```\n{detail_bits[0]}\n```"

    if code in ("validation_error", "invalid_config"):
        st.warning(body)
    elif code == "chunking_config_conflict":
        st.warning(
            body + "\n\n_Change chunk parameters to match existing rows, or use a fresh document._"
        )
    elif code == "unsupported_provider":
        st.warning(body)
    elif code == "no_chunks_to_index":
        st.info(body + "\n\n_Chunk at least one document first._")
    elif code in ("transport_error",):
        st.error(body)
    elif code in ("manifest_not_found", "document_not_found"):
        st.error(
            body
            + "\n\n_Check Postgres is running, migrations applied, and pipeline steps completed._"
        )
    else:
        st.error(body)


def _show_raw_response(title: str, data: dict[str, Any] | None) -> None:
    if data is None:
        return
    with st.expander(title):
        st.code(json.dumps(data, indent=2, default=str), language="json")


def run() -> None:
    st.set_page_config(page_title="legal-rag-engine demo", layout="wide")
    init_state(st)
    st.title("legal-rag-engine demo")
    st.caption("HTTP-only demo over `/v1`; no direct engine imports.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("API")
        base = st.text_input(
            "Base URL",
            value=st.session_state[SessionKeys.base_url],
            key="inp_base_url",
        )
        st.session_state[SessionKeys.base_url] = (
            base.strip() or st.session_state[SessionKeys.base_url]
        )

        if st.button("Check health", use_container_width=True):
            try:
                with _client_for_base(st.session_state[SessionKeys.base_url]) as c:
                    h = c.health()
                st.session_state[SessionKeys.health_status] = "ok"
                st.session_state[SessionKeys.health_detail] = h
                st.success("API reachable")
            except ApiError as e:
                st.session_state[SessionKeys.health_status] = "error"
                st.session_state[SessionKeys.health_detail] = None
                render_api_error(e)

        if st.session_state.get(SessionKeys.health_detail):
            st.json(st.session_state[SessionKeys.health_detail])

        st.divider()
        st.subheader("Pinned IDs")
        st.text_input(
            "document_id",
            value=st.session_state[SessionKeys.selected_document_id],
            key="inp_doc_pin",
        )
        st.session_state[SessionKeys.selected_document_id] = str(
            st.session_state.get("inp_doc_pin", "")
        ).strip()

        st.text_input(
            "manifest_id (retrieve/answer)",
            value=st.session_state[SessionKeys.selected_manifest_id],
            key="inp_manifest_pin",
        )
        st.session_state[SessionKeys.selected_manifest_id] = str(
            st.session_state.get("inp_manifest_pin", "")
        ).strip()

        # Embedding read-back from last index/retrieve
        last_ix = st.session_state.get(SessionKeys.last_index)
        last_rt = st.session_state.get(SessionKeys.last_retrieve)
        st.caption("Last embedding context (from index or retrieve):")
        emb_line = "—"
        if isinstance(last_rt, dict) and last_rt.get("embedding_provider"):
            emb_line = (
                f"{last_rt.get('embedding_provider')} "
                f"dims={last_rt.get('embedding_dimensions')} "
                f"model={last_rt.get('embedding_model')!r}"
            )
        elif isinstance(last_ix, dict) and last_ix.get("embedding_provider"):
            emb_line = (
                f"{last_ix.get('embedding_provider')} "
                f"dims={last_ix.get('embedding_dimensions')} "
                f"model={last_ix.get('embedding_model')!r}"
            )
        st.text(emb_line)

    base_url = st.session_state[SessionKeys.base_url]

    tab_docs, tab_chunk, tab_index, tab_ret, tab_ans = st.tabs(
        ["Documents", "Chunking", "Indexing", "Retrieval", "Answer"]
    )

    # --- Documents ---
    with tab_docs:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("List documents", key="btn_list_docs"):
                try:
                    with _client_for_base(base_url) as c:
                        data = c.list_documents(limit=100, offset=0)
                    st.session_state["lr_last_documents"] = data
                except ApiError as e:
                    render_api_error(e)

            docs_data = st.session_state.get("lr_last_documents")
            if isinstance(docs_data, dict) and docs_data.get("documents"):
                rows = docs_data["documents"]
                ids = [str(d.get("id", "")) for d in rows if d.get("id")]
                labels = [
                    f'{d.get("title") or "—"} — {d.get("id")} — {d.get("source_path", "")}'
                    for d in rows
                ]
                choice = st.selectbox(
                    "Select document",
                    range(len(labels)),
                    format_func=lambda i: labels[i],
                )
                if st.button("Use selected as pinned document_id", key="btn_pin_doc"):
                    st.session_state[SessionKeys.selected_document_id] = ids[choice]
                    st.session_state["inp_doc_pin"] = ids[choice]
                    st.rerun()
            raw_docs = docs_data if isinstance(docs_data, dict) else None
            _show_raw_response("Raw API response (documents list)", raw_docs)

        with col_b:
            ingest_path = st.text_input(
                "Path to file (server-local)",
                value="data/sample_corpus/basic/intro.md",
                key="ingest_path",
            )
            persist = st.checkbox("persist", value=True, key="ingest_persist")
            if st.button("Ingest document", key="btn_ingest"):
                try:
                    with _client_for_base(base_url) as c:
                        data = c.ingest(ingest_path, persist=persist)
                    st.session_state[SessionKeys.last_ingest] = data
                    did = data.get("document_id")
                    if did:
                        st.session_state[SessionKeys.selected_document_id] = str(did)
                        st.session_state["inp_doc_pin"] = str(did)
                    st.success("Ingest completed")
                except ApiError as e:
                    render_api_error(e)
            last_ing = st.session_state.get(SessionKeys.last_ingest)
            if isinstance(last_ing, dict):
                mcols = st.columns(3)
                mcols[0].metric("document_id", str(last_ing.get("document_id", "—"))[:8] + "…")
                mcols[1].metric("checksum", (last_ing.get("checksum") or "—")[:16])
                mcols[2].metric("persisted", str(last_ing.get("persisted")))
                st.write(
                    {
                        "source_path": last_ing.get("source_path"),
                        "title": last_ing.get("title"),
                        "processing_run_id": last_ing.get("processing_run_id"),
                        "created": last_ing.get("created"),
                        "skipped": last_ing.get("skipped"),
                    }
                )
            ing_raw = last_ing if isinstance(last_ing, dict) else None
            _show_raw_response("Raw API response (ingest)", ing_raw)

    # --- Chunking ---
    with tab_chunk:
        doc_for_chunk = st.text_input(
            "document_id",
            value=st.session_state[SessionKeys.selected_document_id],
            key="chunk_doc_id",
        )
        strat = st.selectbox("strategy", CHUNK_STRATEGIES, index=0, key="chunk_strat")
        c1, c2, c3 = st.columns(3)
        with c1:
            chunk_size = st.number_input("chunk_size", min_value=1, value=1200, step=100)
        with c2:
            chunk_overlap = st.number_input("chunk_overlap", min_value=0, value=200, step=10)
        with c3:
            preview_chars = st.number_input("chunk_preview_chars", min_value=1, value=240, step=20)
        preserve = st.checkbox("preserve_headings", value=True, key="chunk_preserve")

        if st.button("Chunk document (include_chunks=true)", key="btn_chunk"):
            if not (doc_for_chunk or "").strip():
                st.error("document_id is required")
            else:
                try:
                    with _client_for_base(base_url) as c:
                        data = c.chunk(
                            document_id=doc_for_chunk.strip(),
                            strategy=strat,
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                            preserve_headings=preserve,
                            include_chunks=True,
                            chunk_preview_chars=int(preview_chars),
                        )
                    st.session_state[SessionKeys.last_chunk] = data
                    if data.get("skipped_existing"):
                        st.info("Chunk run skipped (existing chunks).")
                    else:
                        st.success("Chunking completed")
                except ApiError as e:
                    render_api_error(e)

        last_ch = st.session_state.get(SessionKeys.last_chunk)
        if isinstance(last_ch, dict):
            st.write(
                {
                    "processing_run_id": last_ch.get("processing_run_id"),
                    "strategy": last_ch.get("strategy"),
                    "config_hash": last_ch.get("config_hash"),
                    "created": last_ch.get("created"),
                    "skipped_existing": last_ch.get("skipped_existing"),
                    "chunks_count": last_ch.get("chunks_count"),
                }
            )
            for i, ch in enumerate(last_ch.get("chunks") or []):
                exp_title = (
                    f'Chunk {i + 1} — idx {ch.get("chunk_index")} — {ch.get("chunk_id")}'
                )
                with st.expander(exp_title):
                    st.write(ch)
        ch_raw = last_ch if isinstance(last_ch, dict) else None
        _show_raw_response("Raw API response (chunk)", ch_raw)

        st.divider()
        st.subheader("Existing chunks (GET)")
        gd = st.text_input("document_id for listing", value=doc_for_chunk, key="list_chunks_doc")
        list_strat = st.text_input("filter strategy (optional)", value="", key="list_chunks_strat")
        if st.button("Load persisted chunks", key="btn_list_chunks"):
            if not (gd or "").strip():
                st.error("document_id is required")
            else:
                try:
                    with _client_for_base(base_url) as c:
                        data = c.list_document_chunks(
                            gd.strip(),
                            strategy=list_strat.strip() or None,
                            limit=200,
                        )
                    st.session_state["lr_last_doc_chunks"] = data
                except ApiError as e:
                    render_api_error(e)
        dc = st.session_state.get("lr_last_doc_chunks")
        if isinstance(dc, dict):
            st.write(f"count={dc.get('count')} strategy={dc.get('strategy')}")
            for ch in dc.get("chunks") or []:
                with st.expander(f'Chunk idx {ch.get("chunk_index")} — {ch.get("chunk_id")}'):
                    st.write(ch)
        dc_raw = dc if isinstance(dc, dict) else None
        _show_raw_response("Raw API response (document chunks)", dc_raw)

    # --- Indexing ---
    with tab_index:
        ix_strat = st.text_input("chunking_strategy filter", value="fixed_size", key="ix_strat")
        inc_d = st.checkbox("include_dense", value=True, key="ix_dense")
        inc_s = st.checkbox("include_sparse", value=True, key="ix_sparse")
        if not inc_d and not inc_s:
            st.warning("At least one of dense/sparse must be true (matches API validator).")
        force = st.checkbox("force_reindex", value=False, key="ix_force")
        with st.expander("Optional embedding overrides (leave blank for server Settings)"):
            ov_prov = st.text_input("embedding_provider", value="", key="ix_eprov")
            ov_model = st.text_input("embedding_model", value="", key="ix_emodel")
            st.text_input("embedding_dimensions", value="", key="ix_edim")

        def _optional_int(s: str) -> int | None:
            s = (s or "").strip()
            if not s:
                return None
            return int(s)

        if st.button("Build index", key="btn_index"):
            if not inc_d and not inc_s:
                st.error("Select at least one of include_dense / include_sparse")
            else:
                try:
                    dims = _optional_int(str(st.session_state.get("ix_edim", "")))
                    with _client_for_base(base_url) as c:
                        data = c.index(
                            chunking_strategy=ix_strat.strip() or None,
                            include_dense=inc_d,
                            include_sparse=inc_s,
                            embedding_provider=ov_prov.strip() or None,
                            embedding_model=ov_model.strip() or None,
                            embedding_dimensions=dims,
                            force_reindex=force,
                        )
                    st.session_state[SessionKeys.last_index] = data
                    mid = data.get("manifest_id")
                    if mid:
                        st.session_state[SessionKeys.selected_manifest_id] = str(mid)
                        st.session_state["inp_manifest_pin"] = str(mid)
                    if data.get("skipped_existing"):
                        st.info("Index skipped (existing manifest). Inspect manifest_id below.")
                    else:
                        st.success("Indexing completed")
                except ApiError as e:
                    render_api_error(e)

        last_ix2 = st.session_state.get(SessionKeys.last_index)
        if isinstance(last_ix2, dict):
            st.write(
                {
                    "manifest_id": last_ix2.get("manifest_id"),
                    "manifest_hash": last_ix2.get("manifest_hash"),
                    "config_hash": last_ix2.get("config_hash"),
                    "chunk_set_hash": last_ix2.get("chunk_set_hash"),
                    "chunking_strategy": last_ix2.get("chunking_strategy"),
                    "embedding_provider": last_ix2.get("embedding_provider"),
                    "embedding_model": last_ix2.get("embedding_model"),
                    "embedding_dimensions": last_ix2.get("embedding_dimensions"),
                    "include_dense": last_ix2.get("include_dense"),
                    "include_sparse": last_ix2.get("include_sparse"),
                    "embeddings_persisted": last_ix2.get("embeddings_persisted"),
                    "indexed_chunk_count": last_ix2.get("indexed_chunk_count"),
                    "processing_run_id": last_ix2.get("processing_run_id"),
                    "created": last_ix2.get("created"),
                    "skipped_existing": last_ix2.get("skipped_existing"),
                }
            )
        ix_raw = last_ix2 if isinstance(last_ix2, dict) else None
        _show_raw_response("Raw API response (index)", ix_raw)

        if st.button("Refresh manifest list", key="btn_manifests"):
            try:
                filt = ix_strat.strip() or None
                with _client_for_base(base_url) as c:
                    ml = c.list_manifests(limit=50, chunking_strategy=filt)
                st.session_state[SessionKeys.last_manifests] = ml
            except ApiError as e:
                render_api_error(e)
        ml_data = st.session_state.get(SessionKeys.last_manifests)
        if isinstance(ml_data, dict) and ml_data.get("manifests"):
            for m in ml_data["manifests"]:
                cols = st.columns([4, 1])
                cols[0].write(
                    f'{m.get("manifest_id")} — dense={m.get("include_dense")} '
                    f'sparse={m.get("include_sparse")} — {m.get("indexed_chunk_count")} chunks'
                )
                mid_s = str(m.get("manifest_id", ""))
                if cols[1].button("Pin", key=f"pin_{mid_s}"):
                    st.session_state[SessionKeys.selected_manifest_id] = mid_s
                    st.session_state["inp_manifest_pin"] = mid_s
                    st.rerun()
        ml_raw = ml_data if isinstance(ml_data, dict) else None
        _show_raw_response("Raw API response (manifests list)", ml_raw)

    # --- Retrieval ---
    with tab_ret:
        q = st.text_input("query", value="What does the intro describe?", key="ret_query")
        mode = st.selectbox("mode", RETRIEVAL_MODES, index=0, key="ret_mode")
        top_k = st.number_input("top_k", min_value=1, max_value=50, value=5, key="ret_topk")
        ret_strat = st.text_input("chunking_strategy", value="fixed_size", key="ret_strat")
        use_pin = st.checkbox("Use pinned index_manifest_id", value=True, key="ret_use_pin")
        pin_m = st.session_state[SessionKeys.selected_manifest_id].strip()
        manifest_for_ret: str | None = pin_m if use_pin and pin_m else None

        if mode == "hybrid":
            st.info(
                "Hybrid requires a manifest built with **include_sparse=true**. "
                "Otherwise the API returns `manifest_not_found`."
            )

        meta_filter_ret = _metadata_filter_from_expander("ret")

        if st.button("Retrieve", key="btn_ret"):
            try:
                with _client_for_base(base_url) as c:
                    data = c.retrieve(
                        query=q,
                        mode=mode,
                        chunking_strategy=ret_strat.strip() or None,
                        top_k=int(top_k),
                        index_manifest_id=manifest_for_ret,
                        metadata_filter=meta_filter_ret,
                    )
                st.session_state[SessionKeys.last_retrieve] = data
                st.success(f"Retrieved {data.get('total_results', 0)} result(s)")
            except ApiError as e:
                render_api_error(e)

        lr = st.session_state.get(SessionKeys.last_retrieve)
        if isinstance(lr, dict):
            st.write(
                {
                    "index_manifest_id": lr.get("index_manifest_id"),
                    "manifest_hash": lr.get("manifest_hash"),
                    "embedding_provider": lr.get("embedding_provider"),
                    "embedding_model": lr.get("embedding_model"),
                    "embedding_dimensions": lr.get("embedding_dimensions"),
                    "mode": lr.get("mode"),
                }
            )
            for h in lr.get("chunks") or []:
                rank = h.get("rank")
                with st.expander(
                    f"Rank {rank} — score RRF {h.get('rrf_score')} "
                    f"dense {h.get('dense_score')} sparse {h.get('sparse_score')}"
                ):
                    st.write(
                        {
                            "chunk_id": h.get("chunk_id"),
                            "document_id": h.get("document_id"),
                            "source_path": h.get("source_path"),
                            "title": h.get("title"),
                            "heading": h.get("heading"),
                            "chunk_index": h.get("chunk_index"),
                            "chunking_strategy": h.get("chunking_strategy"),
                            "retrieval_sources": h.get("retrieval_sources"),
                        }
                    )
                    st.text(h.get("text_preview") or "")
        _show_raw_response("Raw API response (retrieve)", lr if isinstance(lr, dict) else None)

    # --- Answer ---
    with tab_ans:
        question = st.text_input(
            "question",
            value="What does the intro describe?",
            key="ans_q",
        )
        amode = st.selectbox("mode", RETRIEVAL_MODES, index=0, key="ans_mode")
        atop_k = st.number_input("top_k", min_value=1, max_value=50, value=5, key="ans_topk")
        astrat = st.text_input("chunking_strategy", value="fixed_size", key="ans_strat")
        st.text_input(
            "provider (API accepts mock only)",
            value="mock",
            disabled=True,
            key="ans_prov",
            help="Only `mock` is supported on the server in this phase.",
        )
        use_pin_a = st.checkbox("Use pinned index_manifest_id", value=True, key="ans_use_pin")
        pin_ma = st.session_state[SessionKeys.selected_manifest_id].strip()
        manifest_for_ans: str | None = pin_ma if use_pin_a and pin_ma else None

        if amode == "hybrid":
            st.info("Hybrid requires sparse-enabled manifest (same as retrieve).")

        meta_filter_ans = _metadata_filter_from_expander("ans")

        if st.button("Generate grounded answer", key="btn_ans"):
            try:
                with _client_for_base(base_url) as c:
                    data = c.answer(
                        question=question,
                        mode=amode,
                        chunking_strategy=astrat.strip() or None,
                        top_k=int(atop_k),
                        provider="mock",
                        index_manifest_id=manifest_for_ans,
                        metadata_filter=meta_filter_ans,
                    )
                st.session_state[SessionKeys.last_answer] = data
                mode_v = data.get("mode")
                if mode_v == "grounded":
                    st.success("Answer mode: **grounded**")
                elif mode_v == "partial":
                    st.warning("Answer mode: **partial**")
                else:
                    st.info("Answer mode: **insufficient_context**")
            except ApiError as e:
                render_api_error(e)

        la = st.session_state.get(SessionKeys.last_answer)
        if isinstance(la, dict):
            st.markdown("### Answer")
            st.markdown(la.get("answer") or "")
            st.write(
                {
                    "mode": la.get("mode"),
                    "insufficient_context": la.get("insufficient_context"),
                    "retrieval_mode": la.get("retrieval_mode"),
                    "used_citation_ids": la.get("used_citation_ids"),
                }
            )
            cver = la.get("citation_verification")
            if isinstance(cver, dict):
                st.markdown("#### Citation verification")
                st.json(cver)
            st.markdown("#### Citations")
            for cite in la.get("citations") or []:
                st.write(cite)
            meta = la.get("metadata") or {}
            if meta.get("citation_validity_rate") is not None:
                st.metric("citation_validity_rate (metadata)", meta.get("citation_validity_rate"))
        _show_raw_response("Raw API response (answer)", la if isinstance(la, dict) else None)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
