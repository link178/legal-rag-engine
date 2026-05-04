# Visión general del sistema

Legal RAG Engine es un repositorio **engine-first**: prioriza un pipeline RAG reproducible, trazable y evaluable. La API y la demo consumen el motor; no definen el producto.

La v1 es **local-first** y mantiene una **zero-cost baseline** para que cualquier persona pueda probar el sistema con un corpus pequeño sin servicios externos de pago.

## Pipeline de alto nivel

```text
Document Sources
    → Loaders / Parsers
    → Normalizer
    → Chunker
    → PostgreSQL documents/chunks
    → pgvector storage (dense index / ANN in later phases)
    → Postgres FTS / sparse local index
    → Dense/Sparse/Hybrid Retriever
    → RRF Fusion
    → Optional Reranker
    → Context Builder
    → Grounded Generator
    → Citation Verifier
    → API / Streamlit Demo / Eval Reports
```

## Componentes principales

- **Loaders / Parsers**: cargan Markdown, TXT, HTML, PDF y corpus tipo `legalize-*`.
- **Normalizer**: limpia texto, conserva metadatos y calcula checksums.
- **Chunker**: produce chunks comparables con estrategia fixed-size o structure-aware.
- **PostgreSQL documents/chunks**: mantiene trazabilidad, metadatos y resultados de evaluación.
- **pgvector**: backend vectorial default de v1 (Phase 4B persiste embeddings; índice ANN y retrieval denso en fases posteriores).
- **Postgres FTS / sparse local index**: rama de sparse retrieval inicial.
- **Dense/Sparse/Hybrid Retriever**: permite modos comparables.
- **RRF Fusion**: combina rankings dense y sparse.
- **Context Builder**: prepara contexto auditable para generación.
- **Grounded Generator**: responde solo con contexto, con mock generation para tests y local generation opcional.
- **Citation Verifier**: comprueba referencias y soporta insufficient context fallback.

## Local-first v1 boundary

- PostgreSQL/pgvector es el default de v1.
- Qdrant queda fuera de la baseline y solo se considera optional adapter futuro.
- No se requiere API externa para smoke tests.
- Embeddings locales son el default recomendado para zero-cost baseline.
- Proveedores externos de embeddings/LLM son optional adapters detrás de provider interfaces.
- Legal/compliance product logic queda fuera del repo público.
- El sistema debe poder ejecutarse con un corpus pequeño de ejemplo.

## Notas de diseño

- La trazabilidad en ingestión y chunking facilita evaluación, depuración y citation verification.
- Hybrid retrieval combina dense retrieval y sparse retrieval sin esconder el core del algoritmo.
- La UI Streamlit y la API FastAPI son consumidores del mismo núcleo, no una plataforma SaaS.

Para decisiones concretas de stack y extensiones opcionales, ver [`../decisions/0001-local-first-zero-cost-stack.md`](../decisions/0001-local-first-zero-cost-stack.md).
