# ADR-0001: Local-first zero-cost baseline

## Status

Accepted

## Context

El proyecto **Legal RAG Engine** debe ser ejecutable por terceros, servir como portfolio técnico y reducir la fricción operativa para contributors, recruiters y usuarios open-source.

La v1 debe evitar coste obligatorio, cuentas cloud o API keys necesarias para smoke tests. También debe dejar espacio para adapters futuros sin acoplar el core a un único proveedor, vector store gestionado o plataforma comercial.

El repositorio público es un motor RAG reusable. No es un SaaS, no es una plataforma enterprise y no debe contener lógica propietaria de compliance legal avanzado.

## Decision

- Use PostgreSQL + pgvector as default vector backend.
- Use Postgres FTS and/or local sparse retrieval for sparse search.
- Keep Qdrant as optional future adapter.
- Use local embeddings by default when possible.
- Support mock generation for tests.
- Support local LLM generation optionally, for example through Ollama.
- Keep external providers optional behind provider interfaces.
- Keep Docker Compose focused on `api`, `postgres` with pgvector, and optional `streamlit`.
- Do not require external API keys for basic tests, smoke tests or the baseline demo.

## Consequences

### Positive

- Lower setup cost.
- Simpler Docker Compose.
- Easier reproducibility.
- Clearer learning path.
- Fewer moving parts.
- Better alignment with an open-source portfolio repo.

### Negative

- Less specialized vector search than Qdrant.
- Local embeddings may be slower.
- Local LLM quality may vary.
- Future adapter abstraction must be designed carefully.
- Some production-scale retrieval features may require optional adapters later.
