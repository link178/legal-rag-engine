# Manual técnico de implementación — Legal RAG Engine

## 1. Objetivo técnico

Implementar un motor RAG híbrido reusable, **engine-first**, **local-first** y con **zero-cost baseline**, centrado en calidad de retrieval, grounding, citation verification, trazabilidad y evaluación reproducible.

El motor debe funcionar sobre corpus documentales generales y corpus legales tipo `legalize-*`, pero el caso legal no convierte el repositorio público en un producto de compliance ni en una plataforma SaaS.

## 2. Stack propuesto v1

### Backend

- Python 3.11+ / 3.12.
- FastAPI.
- Pydantic v2.
- Typer para CLI.

### Persistencia

- PostgreSQL para documentos, chunks, metadatos, resultados de evaluación y configuración mínima.
- pgvector para búsqueda vectorial.
- Postgres Full Text Search para sparse retrieval inicial.
- BM25 local opcional para comparativas o fallback.

### Vector stores

- **Default v1**: pgvector.
- **Optional future adapter**: Qdrant.
- No introducir Qdrant en `docker-compose.yml`, settings principales ni scripts seed de v1.

### Procesamiento documental

- `pymupdf` o `pdfplumber` para PDF.
- `beautifulsoup4` para HTML.
- Parser Markdown nativo/simple.
- Parser de frontmatter para corpus tipo `legalize-*`.

### LLM / embeddings

- Interfaz propia para embeddings.
- Proveedor local por defecto para embeddings cuando sea viable.
- Proveedores externos opcionales detrás de provider interfaces.
- Mock generation para tests.
- Local generation opcional, por ejemplo Ollama, para demo sin coste obligatorio.
- Ningún smoke test debe exigir API keys externas.

### Frontend demo

- Streamlit para demo inicial.

### Infra local

- Docker Compose mínimo con `api`, `postgres` con pgvector y `streamlit` opcional.
- Entorno local reproducible con corpus pequeño de ejemplo.

## 3. Principios de implementación

1. **Engine-first**: el núcleo es el pipeline RAG, no la UI ni un producto SaaS.
2. **Local-first**: la baseline debe correr en una máquina de desarrollo con Docker Compose.
3. **Zero-cost baseline**: demo y tests básicos sin servicios externos de pago.
4. **Framework-light**: no ocultar retrieval, RRF, citas y evaluación dentro de wrappers grandes.
5. **Provider interfaces**: embeddings, generation, reranking y vector stores externos deben quedar detrás de interfaces.
6. **Core explícito**: RRF, context building, citation verification y fallback deben ser entendibles en código propio.
7. **Persistencia clara**: documentos, chunks, embeddings y resultados deben ser trazables.
8. **Sin lógica comercial**: fuera SaaS, AI Act product, scoring regulatorio, multi-tenant, billing y dashboards enterprise.

## 4. Estructura recomendada del repo

```text
legal-rag-engine/
├─ app/
│  ├─ api/
│  │  ├─ routes/
│  │  ├─ schemas/
│  │  └─ dependencies/
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ logging.py
│  │  └─ constants.py
│  ├─ domain/
│  │  ├─ models/
│  │  └─ services/
│  ├─ ingestion/
│  │  ├─ loaders/
│  │  ├─ parsers/
│  │  ├─ normalizers/
│  │  └─ adapters/
│  ├─ chunking/
│  │  ├─ strategies/
│  │  └─ service.py
│  ├─ indexing/
│  │  ├─ dense/
│  │  ├─ sparse/
│  │  └─ sync.py
│  ├─ retrieval/
│  │  ├─ dense.py
│  │  ├─ sparse.py
│  │  ├─ fusion.py
│  │  ├─ rerank.py
│  │  └─ orchestrator.py
│  ├─ generation/
│  │  ├─ prompts/
│  │  ├─ answerer.py
│  │  └─ citations.py
│  ├─ evaluation/
│  │  ├─ datasets/
│  │  ├─ metrics/
│  │  ├─ runners/
│  │  └─ reports/
│  ├─ storage/
│  │  ├─ postgres/
│  │  ├─ pgvector/
│  │  ├─ sparse/
│  │  └─ files/
│  └─ ui/
│     └─ streamlit_app.py
├─ docs/
│  ├─ architecture/
│  ├─ decisions/
│  ├─ implementation/
│  ├─ learning/
│  └─ product/
├─ scripts/
├─ tests/
├─ data/
│  ├─ sample_corpus/
│  └─ eval/
├─ docker/
├─ docker-compose.yml
├─ pyproject.toml
└─ README.md
```

Qdrant no aparece en la estructura principal de v1. Si se añade más adelante, debe entrar como optional adapter separado y documentado.

## 5. Modelo de dominio

### Document

Representa el documento original. Campos sugeridos: `id`, `source_path`, `source_type`, `title`, `raw_text`, `normalized_text`, `metadata_json`, `checksum`, `created_at`.

### Chunk

Representa un fragmento indexable. Campos sugeridos: `id`, `document_id`, `chunk_index`, `text`, `heading`, `page_number`, `chunking_strategy`, `char_count`, `token_estimate`, `metadata_json`, `embedding` o referencia a embedding persistido, y datos necesarios para FTS/sparse retrieval.

### RetrievalResult

Resultado recuperado antes o después de fusión/reranking. Campos sugeridos: `chunk_id`, `retrieval_mode`, `dense_score`, `sparse_score`, `fusion_score`, `rerank_score`, `rank_position`.

### Answer

Respuesta generada. Campos sugeridos: `question`, `answer_text`, `citations`, `confidence`, `insufficient_context`, `retrieved_chunk_ids`, `generation_metadata`.

### EvalCase

Caso de evaluación. Campos sugeridos: `id`, `question`, `expected_answer`, `expected_sources`, `case_type`, `tags`.

## 6. Pipeline técnico completo

```text
Document Sources
    → Loaders / Parsers
    → Normalizer
    → Chunker
    → PostgreSQL documents/chunks
    → pgvector dense index
    → Postgres FTS / sparse local index
    → Dense/Sparse/Hybrid Retriever
    → RRF Fusion
    → Optional Reranker
    → Context Builder
    → Grounded Generator
    → Citation Verifier
    → API / Streamlit Demo / Eval Reports
```

## 7. Ingestión, parsing y normalización

Entrada: directorio local, lista de archivos, corpus de ejemplo o adaptador `legalize-*`.

Flujo:

1. Detectar tipo de fichero.
2. Cargar contenido.
3. Extraer metadatos.
4. Normalizar texto.
5. Calcular checksum.
6. Persistir documento.
7. Enviar a chunking.

Recomendaciones:

- Usar checksum para evitar reprocesado innecesario.
- Almacenar raw y normalized text.
- Registrar errores por archivo sin abortar toda la ingestión.
- Conservar metadatos legales cuando existan, sin introducir lógica de compliance avanzada.

## 8. Chunking

Implementar una interfaz común:

```python
class ChunkingStrategy(Protocol):
    def split(self, document: Document, config: ChunkingConfig) -> list[Chunk]: ...
```

Baseline fixed-size + Markdown structure-aware implementations live under `app/chunking/`; persistence/idempotency semantics are documented in [`chunking-notes.md`](chunking-notes.md).

## 9. Embeddings e indexado dense

**Implemented through Phase 4B of the repo público:** interfaz `EmbeddingProvider`, proveedor determinista, proveedor opcional `local_sentence_transformers`, sparse term-frequency inicial, tablas `index_manifests`/`index_manifest_chunks`, tabla **`chunk_embeddings`** (pgvector), CLI `python -m app.indexing.cli`. Índices ANN sobre vectores y FTS en Postgres siguen como trabajo opcional; ver [`indexing-notes.md`](indexing-notes.md).

**Phase 5 (retrieval)** añade consulta de similaridad denso (L2 pgvector), sparse léxico baseline sobre `sparse_terms_json`, modo híbrido con RRF y CLI `python -m app.retrieval.cli`; ver [`retrieval-notes.md`](retrieval-notes.md).

**Phase 5.5 (evaluación retrieval-only baseline):** dataset golden en JSONL, métricas Hit@k / MRR, informes JSON y Markdown opcional, CLI `python -m app.evaluation.cli`; lectura únicamente (sin nuevas filas `processing_runs` para eval). Ver [`evaluation-notes.md`](evaluation-notes.md).

**Phase 6 (generation mock wired):** `ContextBuilder`, `build_grounded_prompt`, protocolo `GenerationProvider` (`generate(prompt: str) -> str`), `MockGenerationProvider`, `GroundedAnswerer.from_session`, CLI `python -m app.generation.cli`; lectura únicamente PostgreSQL (**sin nuevas filas `processing_runs`**). Ver [`generation-notes.md`](generation-notes.md).

**Phase 7 (citation verification):** `verify_citations`, `CitationVerificationResult`, integración en `GroundedAnswerer` y CLI JSON/resumen humano (`citation_verification`); sin migraciones ni tablas nuevas.

Provider interface:

```python
class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Default v1:

- Usar proveedor local cuando sea viable para mantener la zero-cost baseline.
- Persistir embeddings en PostgreSQL con pgvector.
- Indexar por `chunk_id` y conservar vínculo con `document_id` y metadatos.
- Soportar batching.

Optional adapters:

- APIs externas de embeddings pueden añadirse detrás de la interfaz.
- Qdrant puede añadirse en el futuro como vector store adapter, pero no como dependencia default.

## 10. Indexado sparse

Default v1:

- Postgres Full Text Search sobre chunks persistidos.
- Configuración reproducible de idioma/tokenización cuando aplique.
- Índices GIN para consultas full-text.

BM25 local puede existir para comparativas, tests o fallback, siempre sin convertirse en servicio externo obligatorio.

Dense index y sparse index deben referenciar exactamente el mismo conjunto de `chunk_id`.

Manifiesto recomendado: versión del corpus, estrategia de chunking, checksums de documentos, número de chunks, proveedor/modelo de embeddings y fecha de construcción.

## 11. Retrieval

### Dense retrieval

1. Embed query mediante `EmbeddingProvider`.
2. Consultar pgvector.
3. Recuperar top-k chunks.
4. Enriquecer con metadata desde PostgreSQL.

### Sparse retrieval

1. Tokenizar/normalizar query.
2. Ejecutar Postgres FTS o BM25 local.
3. Recuperar top-k chunks.
4. Enriquecer con metadata.

### Fusión con RRF

Implementación propia:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

El orquestador debe soportar `dense_only`, `sparse_only` y `hybrid`.

## 12. Reranking

Debe ser opcional y quedar detrás de interfaz:

```python
class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk]) -> list[ScoredChunk]: ...
```

Si no está disponible, el sistema debe seguir funcionando.

## 13. Grounded generation (Phase 6 shipped) + citation verification (Phase 7)

Implementado como capa engine-only (sin endpoints FastAPI todavía). Flujo:

1. `GroundedAnswerer.from_session(session, RetrievalConfig(...))`: resuelve el mismo manifest que retrieval y ejecuta `RetrievalOrchestrator.retrieve`.
2. `ContextBuilder` convierte ``RetrievedChunk`` rankeados en `GroundedContextBlock` (ids citables desde 1, truncado determinista).
3. `build_grounded_prompt(question, blocks, insufficient_sentence=...)` genera una sola cadena lista para enviar al proveedor.
4. ``GenerationProvider.generate(prompt: str) -> str`` produce la respuesta en bruto (`MockGenerationProvider` en esta fase).
5. `verify_citations(answer_text, blocks)` (Phase 7) compara todas las apariciones `[N]` con los ids de bloque; clasifica `CitationVerificationResult`.
6. Modo de respuesta: `insufficient_context` si no hay bloques tras `ContextBuilder` o la respuesta igual a `INSUFFICIENT_CONTEXT_SENTENCE`; `grounded` si hay al menos una cita válida y ninguna inválida; `partial` en el resto de casos con contexto (sin citas válidas, sólo inválidas, o mezcla válida+inválida). `used_citation_ids` lleva sólo ids válidos citados.

Formato de contexto determinista (cabeceras de una línea + cuerpo de texto):

```text
[N] source: docs/a.md | title: Intro | heading: Overview
<texto del fragmento truncado opcionalmente con … >
```

Cabeceras con metadatos faltantes usan marcador Unicode `—` para mantener el parseo estable.

Los proveedores v1 efectivos aquí solo incluyen **`MockGenerationProvider`**; proveedores locales/cloud son extensiones opcionales futuras.

## 14. Verificación mecánica de citas y fallback insufficient context

La verificación **implementada** comprueba que cada bracket `[N]` en la respuesta apunte a un bloque de contexto presente en el prompt (sin DB, sin LLM). `GroundedAnswer.citation_verification` y CLI JSON exponen válidas, inválidas, no usadas, duplicados y `citation_validity_rate`. No hay solapamiento léxico ni entailment semántico en esta fase.

`insufficient_context` se sigue activando sólo cuando no hay bloques utilizables después de `ContextBuilder` o cuando la salida es exactamente la frase sentinel; **no** se baja por score de retrieval ni contradicción entre chunks (esto queda fuera del alcance público actual).

## 15. API (Phases 8–9 shipped baseline)

Minimal **engine-aligned** HTTP surface under `/v1` (thin FastAPI routers + Pydantic + error envelope). See [`api-notes.md`](implementation/api-notes.md).

- `POST /v1/ingest`: one local file path; optional `persist` (default true). Wraps `IngestionService` / `ingest_file_persisted` (same as CLI).
- `GET /v1/documents`: list persisted documents (`limit`, `offset`).
- `GET /v1/documents/{document_id}`: metadata + `chunks_count`.
- `GET /v1/documents/{document_id}/chunks`: paginated chunks (optional `strategy`; read-only DB).
- `POST /v1/chunk`: wraps `chunk_document_persisted` (same as `app.chunking.cli`); records `processing_runs` + `chunks`.
- `POST /v1/index`: wraps `index_chunks_persisted` (same as `app.indexing.cli` embedding defaults from `Settings` when omitted); records `processing_runs` + manifests.
- `GET /v1/index-manifests` / `GET /v1/index-manifests/{manifest_id}`: manifest list/detail (read-only DB).
- `POST /v1/retrieve`: body mirrors Phase 5 CLI (`dense_only` | `sparse_only` | `hybrid`, `top_k`, manifest pin `index_manifest_id`, `chunking_strategy`, etc.); embedding family from **server** `Settings`. Read-only DB (no new `processing_runs`).
- `POST /v1/answer`: same retrieval parameters + `question` + `provider` (**`mock` only**, validated before DB session). Returns `mode`, `retrieval_mode`, `citations`, `citation_verification`, `metadata` (same shape as `app.generation.cli` JSON). Read-only DB.

**Still not in scope:** `POST /v1/ask` (single combined endpoint), `POST /v1/evaluate`, multipart upload, streaming, external LLMs, auth.

- `GET /health`: service liveness (Phase 1).

## 16. Evaluación

Golden dataset pequeño, por ejemplo 25-40 preguntas: lookup directo, multi-hop simple, pregunta sin respuesta, pregunta ambigua, término exacto técnico y pregunta dependiente de sección.

Métricas mínimas:

- retrieval hit@k;
- MRR o nDCG si aplica;
- citation validity rate;
- no-answer correctness;
- latencia media.

Comparativas obligatorias:

- dense-only vs hybrid;
- sparse-only vs hybrid;
- fixed-size vs structure-aware;
- mock generation con evaluación retrieval-only durante fases tempranas.

## 17. Demo UI

Con Streamlit en v1. Debe mostrar pregunta, respuesta con citas, chunks recuperados, scores, modo de retrieval y selector de chunking strategy si hay varios índices cargados.

No intentar construir una UI enterprise.

## 18. Configuración

Centralizar en `config.py` o `settings.py`:

- `database_url`;
- provider de embeddings;
- modelo local de embeddings;
- provider generativo (`mock`, `local`, `external_optional`);
- endpoint local opcional para Ollama;
- chunk size;
- overlap;
- top-k;
- RRF k;
- pesos de fusion;
- threshold fallback;
- rerank on/off.

No incluir `qdrant_host` en la configuración principal de v1. Si Qdrant se añade después, debe vivir en configuración de adapter opcional.

Añadir `.env.example` y perfiles `dev`/`test` sin secretos obligatorios.

## 19. Testing

Unit tests: parsers, normalizers, chunkers, RRF, citation parser y fallback logic.

Integration tests: ingest → index → ask, dense retrieval con pgvector, sparse retrieval con Postgres FTS o local, hybrid retrieval end-to-end y eval runner.

Smoke tests: levantar API, consultar corpus de ejemplo y generar respuesta con citas usando mock generation, sin API keys externas.

## 20. Dockerización

`docker-compose` mínimo:

- `api`;
- `postgres` con extensión pgvector;
- `streamlit` opcional.

Opcionalmente un servicio `seed` que cargue corpus de ejemplo, construya índices y deje el sistema listo para demo.

No incluir Qdrant por defecto.

## 21. Plan de implementación por fases

1. **Bootstrap**: estructura repo, config, logging, modelos base y Docker Compose mínimo con Postgres/pgvector.
2. **Ingestión**: loaders markdown/txt/html/pdf, normalización y persistencia documento.
3. **Chunking**: fixed-size, structure-aware y persistencia chunks.
4. **Indexado dense/sparse**: provider local de embeddings, pgvector, Postgres FTS, BM25 local opcional y manifest.
5. **Retrieval híbrido**: dense, sparse, RRF y modo dense-only/sparse-only/hybrid.
6. **Grounded generation**: context builder, mock generation, local generation opcional, prompt grounded, citas y fallback.
7. **Evaluación**: dataset, métricas, comparativas y reporte.
8. **Demo y polish**: Streamlit, README, scripts seed y smoke tests.

## 22. Definition of Done

La implementación pública estará cerrada cuando:

1. El repo levante localmente con Docker Compose.
2. El corpus de ejemplo pueda indexarse con dos estrategias de chunking.
3. PostgreSQL + pgvector sea el vector backend default.
4. Postgres FTS y/o sparse retrieval local cubra la rama sparse.
5. Existan dense-only, sparse-only y hybrid retrieval.
6. La respuesta incluya citas funcionales.
7. El fallback sin contexto funcione.
8. El benchmark corra de extremo a extremo.
9. La demo muestre respuesta, citas y chunks recuperados.
10. Smoke tests no requieran servicios externos de pago.
11. El README explique claramente alcance y exclusiones.

## 23. Frontera pública

Este manual implementa solo el **engine público**.

Quedan fuera: taxonomía AI Act propietaria, motor de impacto regulatorio, scoring regulatorio comercial, evidence packs premium, workflows multi-tenant, usuarios/organizaciones, billing, SSO, conectores enterprise, dashboards comerciales y due diligence avanzada.

Una futura capa privada podría reutilizar el engine, pero debe vivir fuera de la baseline pública.
