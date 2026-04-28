# PRD — Legal RAG Engine

## 1. Resumen ejecutivo

**legal-rag-engine** es un repositorio público open-source, **engine-first**, **local-first** y orientado a portfolio técnico.

El objetivo es construir un motor RAG híbrido, reusable y demostrable sobre corpus documentales generales y corpus legales versionados tipo `legalize-*`, sin convertir el repo público en un producto final de compliance, una plataforma AI Act ni un SaaS.

El proyecto debe demostrar:

- ingestión documental trazable;
- normalización reproducible;
- chunking comparable;
- dense retrieval;
- sparse retrieval;
- hybrid retrieval mediante **RRF**;
- grounded generation con citas;
- citation verification;
- insufficient context fallback;
- evaluación reproducible;
- arquitectura limpia.

## 2. Qué es y qué no es

### Qué es

Un **motor RAG técnico reusable** que expone un pipeline claro para ingesta, indexado, retrieval, generación grounded, citas y evaluación.

### Qué no es

No es un producto completo de compliance, no es un SaaS multi-tenant, no es una herramienta AI Act end-to-end, no es un dashboard enterprise y no incluye lógica comercial cerrada.

## 3. Contexto estratégico

El repo público debe servir como:

1. **Portfolio técnico de alto nivel**: señal clara de ingeniería RAG seria.
2. **Base open-source reusable**: motor aplicable a corpus documentales generales, no solo legales.
3. **Núcleo técnico futuro**: base posible para una capa privada posterior, sin mezclarla con el repositorio público.

La decisión estratégica es mantener pública la infraestructura técnica reusable y dejar fuera cualquier capa que concentre valor comercial específico.

## 4. Problema que resuelve

Muchas demos RAG públicas fallan en:

- retrieval pobre o solo vectorial;
- ausencia de sparse retrieval para términos exactos;
- malas decisiones de chunking;
- falta de trazabilidad;
- citas decorativas o no verificables;
- evaluación inexistente;
- incapacidad para manejar el caso “no hay suficiente contexto”.

Este proyecto cubre ese hueco con una baseline local-first, reproducible y suficientemente concreta para implementación fase a fase.

## 5. Objetivos

### Objetivos principales

1. Permitir indexar un corpus documental local.
2. Persistir documentos, chunks y metadatos de forma trazable.
3. Comparar estrategias de chunking.
4. Construir dense retrieval sobre **PostgreSQL + pgvector** como default v1.
5. Construir sparse retrieval con **Postgres Full Text Search** y/o implementación local.
6. Combinar dense + sparse mediante **RRF**.
7. Responder con grounded generation, citas y fallback por contexto insuficiente.
8. Incluir evaluación reproducible por CLI/API.
9. Exponer una API FastAPI y una demo Streamlit simple.
10. Mantener la demo y los tests básicos ejecutables sin servicios externos de pago.

### Objetivos secundarios

1. Ofrecer adaptadores para corpus legales tipo `legalize-*`.
2. Mantener provider interfaces para embeddings, generation y adapters futuros.
3. Permitir proveedores externos como extensiones opcionales, nunca como requisito de v1.
4. Dejar una ruta limpia hacia adapters futuros como Qdrant sin hacerlo parte de la baseline.

## 6. No objetivos

Este repo público **no** intentará resolver en v1:

- taxonomía AI Act propietaria;
- clasificación completa AI Act;
- scoring regulatorio comercial;
- mapping obligación → control → evidencia;
- evidence packs comerciales;
- workflows multi-tenant;
- usuarios, organizaciones o roles;
- billing;
- SSO;
- dashboards enterprise;
- conectores privados;
- flujos de due diligence avanzada;
- plataforma SaaS completa;
- producto final de compliance legal.

## 7. Usuarios objetivo

- **Recruiter / hiring manager técnico**: busca señal de arquitectura, criterio de ingeniería y ejecución limpia.
- **Developer / OSS user**: quiere reutilizar el engine con su propio corpus documental.
- **Founder / evaluador de producto**: quiere validar si la base técnica podría convertirse después en producto vertical.
- **Usuario demo**: quiere levantar el proyecto localmente, hacer preguntas y ver respuestas con citas.

## 8. Casos de uso principales

1. Indexar documentación técnica o legal.
2. Hacer preguntas con respuestas grounded y citas.
3. Comparar dense-only vs hybrid retrieval.
4. Ejecutar benchmarks y evaluación reproducible.
5. Usar un subconjunto de corpus `legalize-*` como caso legal trazable.

## 9. Alcance funcional v1

### 9.1 Ingestión documental

Debe soportar inicialmente Markdown, TXT, HTML y PDF.

Debe extraer y normalizar texto limpio, fuente, path, título, headings, número de página cuando sea posible, metadatos adicionales y checksum.

### 9.2 Chunking configurable

Se soportarán inicialmente:

1. **Fixed-size with overlap**.
2. **Structure-aware by headings/sections**.

Cada chunk debe registrar `chunk_id`, `document_id`, `chunk_index`, `chunking_strategy`, `char_count`, `token_estimate`, `heading`, `page` y metadatos heredados.

### 9.3 Indexado dense/sparse

El sistema debe construir en paralelo:

- índice dense en **pgvector**;
- índice sparse con **Postgres FTS** y/o sparse retrieval local;
- metadatos por chunk en PostgreSQL;
- manifiesto reproducible de indexado.

**Qdrant no es vector store recomendado por defecto en v1**. Queda documentado solo como **optional adapter** futuro.

### 9.4 Retrieval

Debe soportar dense retrieval, sparse retrieval, hybrid retrieval mediante RRF, top-k configurable y pesos/configuración de fusión cuando aplique.

### 9.5 Reranking

Debe existir una interfaz de reranker opcional. La v1 no debe depender de un reranker externo para funcionar.

### 9.6 Grounded generation

La respuesta debe basarse solo en el contexto recuperado, citar chunks, activar insufficient context fallback cuando no haya soporte suficiente y devolver metadata de citas y fuentes.

La v1 debe soportar **mock generation** para tests y generación local opcional, por ejemplo Ollama, para demo sin coste obligatorio.

### 9.7 Citation verification

El sistema debe verificar que las citas existen, apuntan a chunks recuperados y no están fuera de rango. No hace falta verificación jurídica avanzada en v1.

### 9.8 Evaluación

Debe incluir golden set pequeño, métricas de retrieval, evaluación retrieval-only válida en fases tempranas, comparación dense-only vs hybrid, comparación entre chunking strategies y reportes Markdown/JSON.

### 9.9 API y demo

FastAPI con endpoints mínimos:

- `POST /v1/ingest`;
- `GET /v1/documents`;
- `POST /v1/ask`;
- `POST /v1/evaluate`;
- `GET /health`.

Streamlit debe ofrecer una demo sencilla con pregunta, respuesta, citas, chunks recuperados, scores y switch hybrid vs dense-only.

## 10. Requisitos funcionales

- **RF-1**: ingerir un directorio local de documentos.
- **RF-2**: persistir documentos procesados y chunks para reindexado reproducible.
- **RF-3**: permitir reindexar sin volver a subir el corpus.
- **RF-4**: construir y consultar índices dense y sparse sincronizados.
- **RF-5**: devolver resultados con trazabilidad a documento/chunk.
- **RF-6**: responder con citas inline.
- **RF-7**: manejar el caso “no hay suficiente contexto”.
- **RF-8**: permitir evaluación reproducible por CLI o API.
- **RF-9**: incluir al menos un adaptador básico para corpus `legalize-*`.
- **RF-10**: ejecutarse localmente con Docker Compose.

## 11. Requisitos no funcionales

- **RNF-1 — Reproducibilidad**: la indexación debe ser determinista en la medida de lo posible.
- **RNF-2 — Observabilidad**: loguear ingestión, chunking, indexado, retrieval, generation y evaluación.
- **RNF-3 — Configurabilidad**: exponer chunk size, overlap, top-k, RRF, providers, reranker y thresholds.
- **RNF-4 — Modularidad**: desacoplar ingest, chunking, retrieval, generation, citation verification y evaluation.
- **RNF-5 — Coste**: **La demo y los tests básicos deben poder ejecutarse sin depender obligatoriamente de servicios externos de pago.**
- **RNF-6 — Legibilidad**: el repo debe poder leerse como documentación técnica y portfolio.

## 12. Stack default v1

- Python 3.11+ o 3.12.
- FastAPI.
- Pydantic v2.
- Typer para CLI.
- PostgreSQL para documentos, chunks, metadatos, configuración mínima y resultados de evaluación.
- pgvector como vector backend default.
- Postgres FTS y/o sparse retrieval local para sparse search.
- Streamlit para demo inicial.
- Docker Compose para entorno local.
- Embeddings locales como default recomendado para zero-cost baseline.
- Mock generation para tests.
- Local generation opcional, por ejemplo Ollama.
- Proveedores externos de embeddings/LLM solo como optional adapters detrás de provider interfaces.

## 13. Arquitectura conceptual

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

## 14. Decisiones de diseño

- **Engine-first, no app-first**: API y demo son consumidores del engine.
- **Local-first y zero-cost baseline**: v1 debe probarse con corpus pequeño sin cuentas cloud ni API keys obligatorias.
- **PostgreSQL + pgvector como default v1**: Postgres cubre persistencia, pgvector dense retrieval y FTS/sparse local.
- **Qdrant como optional adapter futuro**: no entra en Docker Compose ni configuración principal de v1.
- **Corpus legal como caso de uso, no límite del engine**: el sistema debe seguir siendo reusable para corpus generales.
- **Evaluación como parte del producto técnico**: no es un script secundario.
- **Sin lógica comercial vertical**: nada de taxonomías propietarias, scoring regulatorio, evidence packs ni workflows enterprise.

## 15. Roadmap v1

1. **Bootstrap**: estructura, configuración, logging, modelos base y Docker Compose mínimo.
2. **Ingestión**: loaders, parsers, normalización y persistencia de documentos.
3. **Chunking**: fixed-size, structure-aware y persistencia de chunks.
4. **Indexado dense/sparse**: embeddings locales, pgvector, Postgres FTS/sparse local y manifiesto de índice.
5. **Retrieval híbrido**: dense, sparse, RRF y modos dense-only/sparse-only/hybrid.
6. **Grounded generation**: context builder, mock generation, generación local opcional, citas y fallback.
7. **Evaluación**: golden set, métricas, comparativas y reportes.
8. **Demo y polish**: Streamlit, README, scripts seed y smoke tests.

## 16. Criterios de aceptación v1

Se considera v1 completada cuando:

1. El repo puede levantarse localmente con instrucciones claras.
2. Puede indexar un corpus pequeño de ejemplo y responder preguntas.
3. Usa PostgreSQL + pgvector como vector backend default.
4. Usa Postgres FTS y/o sparse retrieval local para sparse search.
5. Devuelve respuestas con citas y fuentes trazables.
6. Permite dense-only, sparse-only y hybrid retrieval.
7. Tiene al menos una comparación reproducible en evaluación.
8. Dispone de API funcional y demo mínima.
9. Smoke tests y demo básica no exigen servicios externos de pago.
10. El README deja claro qué incluye y qué queda excluido.

## 17. Exclusiones explícitas

Quedan fuera de este repo público:

- taxonomía AI Act propietaria;
- scoring regulatorio comercial;
- rules engine de impact assessment;
- evidence packs comerciales;
- workflows multi-tenant;
- usuarios/organizaciones;
- billing;
- SSO;
- dashboards enterprise;
- conectores privados;
- plantillas premium;
- lógica de due diligence avanzada;
- producto SaaS completo;
- promesas de compliance legal avanzado.

## 18. Entregables del repo público

- Código fuente del engine.
- Configuración por entorno.
- Documentación técnica y didáctica.
- README de landing.
- Corpus de ejemplo pequeño.
- Golden eval dataset pequeño.
- Scripts de ingestión, indexado y evaluación.
- Docker Compose local-first.
- Demo Streamlit reproducible.
