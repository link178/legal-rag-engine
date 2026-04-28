# Guía paso a paso, detallada y orientada al aprendizaje: cómo construir un RAG profesional desde cero

## Objetivo real de esta guía

Esta guía no está pensada solo para que “montes un proyecto que funcione”. Está pensada para que, al terminar, entiendas con criterio profesional:

- qué es un sistema RAG y qué problemas resuelve realmente;
- qué decisiones de arquitectura importan de verdad;
- cómo se diseña un pipeline de ingestión, chunking, indexado, retrieval y generación grounded;
- qué trade-offs existen en cada decisión;
- cómo evaluar si un RAG funciona bien o solo parece funcionar bien;
- qué diferencia a una demo de un sistema serio y auditable.

La meta final no es solo tener un repositorio bonito. La meta es que seas capaz de diseñar, criticar y construir sistemas RAG con mentalidad de ingeniero.

---

# 0. Qué es RAG, qué no es, y por qué existe

## Definición práctica

RAG significa **Retrieval-Augmented Generation**.

En términos prácticos, es una arquitectura en la que un modelo generativo no responde solo con lo que “recuerda” de su entrenamiento, sino también con información **recuperada en tiempo real** desde un corpus externo.

El patrón básico es:

1. el usuario hace una pregunta;
2. el sistema busca fragmentos relevantes en una base documental;
3. el sistema construye un contexto con esos fragmentos;
4. el LLM genera la respuesta usando ese contexto;
5. idealmente, la respuesta incluye trazabilidad o citas.

## Qué problema resuelve

Un LLM puro tiene varios límites:

- no conoce tus documentos privados;
- puede estar desactualizado;
- puede alucinar;
- no puede citar bien si no tiene una fuente controlada;
- no te permite auditar de dónde sale una respuesta.

RAG existe para reducir esos problemas.

## Qué problema NO resuelve automáticamente

RAG no garantiza por sí solo:

- veracidad;
- ausencia total de alucinaciones;
- respuestas completas;
- buen recall;
- calidad de recuperación;
- cumplimiento legal o trazabilidad real.

Un RAG mal diseñado puede seguir fallando de muchas formas:

- recuperar chunks irrelevantes;
- trocear mal los documentos;
- perder contexto esencial;
- recuperar bien pero sintetizar mal;
- citar fuentes que no soportan la afirmación;
- responder con exceso de confianza.

## Idea mental correcta

Piensa en RAG como un sistema de dos motores:

- un motor de **búsqueda / recuperación**;
- un motor de **síntesis / redacción**.

Si el primero falla, el segundo no puede salvar el resultado.

En sistemas profesionales, el retrieval suele ser más importante que el prompt bonito.

---

# 1. Qué vamos a construir exactamente

Vamos a construir un **motor RAG público, reusable, engine-first y auditado**, centrado en documentación legal/versionada pero suficientemente genérico como para usarse con otros corpus documentales. La guía enseña conceptos profesionales de RAG, no solo instrucciones de implementación.

## Objetivos del proyecto

El sistema deberá poder:

- ingerir documentos en varios formatos;
- normalizarlos y guardarlos con metadatos;
- dividirlos en chunks con estrategias configurables;
- indexarlos con búsqueda híbrida:
  - semántica (dense / embeddings)
  - léxica (sparse / full-text o BM25)
- combinar resultados con Reciprocal Rank Fusion;
- opcionalmente rerankear los mejores resultados;
- generar respuestas grounded con citas;
- indicar cuándo no tiene suficiente información;
- exponer una API con FastAPI;
- tener una demo reproducible;
- incluir evaluación y métricas.

> Nota de alcance v1: la v1 prioriza una baseline local-first y sin coste obligatorio. Las integraciones con proveedores externos quedan detrás de interfaces y no son necesarias para entender ni probar el sistema base.

## Qué NO vamos a construir en esta versión

No construiremos todavía:

- un producto SaaS multi-tenant;
- workflows de compliance;
- un motor comercial de impacto regulatorio;
- un evidence pack premium;
- dashboards enterprise;
- una ontología legal propietaria avanzada.

La razón es importante: primero vamos a entender el **motor técnico de RAG**. La capa de producto viene después.

---

# 2. Principios de diseño que vamos a seguir

Antes de hablar de código, necesitamos principios.

## 2.1. Framework-light

No vamos a delegar toda la lógica en un framework enorme “porque sí”.

Podemos usar librerías útiles para piezas concretas, pero el flujo principal debe quedar en nuestras manos.

### Por qué

Si usas un framework muy abstracto desde el principio:

- aprendes menos;
- entiendes peor los fallos;
- te cuesta depurar;
- no sabes qué parte del sistema realmente aporta valor.

### Qué sí haremos

Usaremos herramientas cuando aporten velocidad o ergonomía, por ejemplo:

- loaders para documentos;
- splitters básicos;
- clientes de embeddings;
- librerías de evaluación.

Pero la lógica crítica debe ser tuya:

- chunking strategy selection;
- metadatos por chunk;
- retrieval híbrido;
- fusión;
- reranking;
- construcción del contexto;
- formateo de citas;
- evaluación.

## 2.2. Source-grounded by design

Todo debe estar trazado.

Cada chunk debe saber:

- de qué documento viene;
- qué posición ocupa;
- qué heading o sección tenía;
- qué estrategia de chunking lo creó;
- qué texto exacto contiene;
- si pertenece a una página o artículo concreto.

### Por qué

Sin trazabilidad:

- no puedes auditar;
- no puedes depurar retrieval;
- no puedes verificar citas;
- no puedes mejorar el sistema con datos.

## 2.3. Configurable, pero no caótico

Tendremos configurabilidad, pero no una explosión de switches sin criterio.

Ejemplos razonables:

- tamaño de chunk;
- overlap;
- top-k retrieval;
- pesos dense vs sparse;
- reranking on/off;
- modelo de embeddings;
- prompt template.

Lo importante es que cada parámetro exista por una razón clara.

## 2.4. Medir antes de opinar

En RAG es muy fácil autoengañarse.

Un sistema puede “parecer bueno” en 5 preguntas manuales y ser mediocre en un conjunto real.

Por eso desde el principio asumimos que necesitaremos:

- dataset de evaluación;
- métricas;
- comparativas;
- logs de fallos.

---

# 3. Arquitectura conceptual completa

Antes de implementar, conviene visualizar el flujo entero.

## Pipeline de alto nivel

1. **Ingestión**
   - cargar documentos
   - detectar formato
   - extraer texto
   - extraer metadatos
   - calcular checksum

2. **Normalización**
   - limpiar texto
   - estandarizar saltos de línea
   - preservar headings relevantes
   - adaptar formatos como `legalize-*`

3. **Chunking**
   - dividir el documento en fragmentos
   - registrar origen y metadatos

4. **Indexado**
   - generar embeddings
   - guardar vectores
   - construir índice sparse
   - persistir chunks en base de datos

5. **Query-time retrieval**
   - procesar pregunta
   - dense retrieval
   - sparse retrieval
   - fusionar resultados
   - rerankear candidatos

6. **Context assembly**
   - seleccionar chunks finales
   - numerarlos
   - construir contexto para el LLM

7. **Generation**
   - prompt grounded
   - respuesta con citas
   - manejo de “no suficiente información”

8. **Post-processing**
   - parsear citas
   - verificar referencias válidas
   - calcular scores de confianza

9. **Evaluation & observability**
   - medir retrieval
   - medir calidad de respuesta
   - comparar configuraciones

---

# 4. Modelado del dominio

Un error muy frecuente es pensar en RAG como “una cadena de funciones”.

A nivel profesional conviene pensar en **entidades**.

## 4.1. Document

Representa el archivo o recurso fuente.

Campos útiles:

- `id`
- `source_path`
- `source_type` (`pdf`, `md`, `html`, `txt`)
- `title`
- `checksum`
- `created_at`
- `updated_at`
- `metadata_json`
- `normalized_text`
- `status`

### Por qué existe

Te permite:

- reindexar sin perder origen;
- detectar cambios;
- auditar el corpus;
- relacionar chunks con el documento madre.

## 4.2. Chunk

Representa una unidad indexable y recuperable.

Campos útiles:

- `id`
- `document_id`
- `chunk_index`
- `text`
- `char_count`
- `token_estimate`
- `heading`
- `page_number`
- `section_path`
- `chunking_strategy`
- `checksum`
- `embedding`
- `tsvector` o referencia al índice sparse

### Por qué existe

Porque el chunk es la verdadera unidad operativa del RAG.

## 4.3. RetrievalResult

Representa un resultado recuperado en query-time.

Campos útiles:

- `chunk_id`
- `dense_score`
- `sparse_score`
- `rrf_score`
- `rerank_score`
- `final_rank`
- `retrieval_sources` (`dense`, `sparse`, `both`)

### Por qué existe

Te permite entender por qué un chunk llegó al top final.

## 4.4. Answer

Representa la respuesta generada.

Campos útiles:

- `question`
- `answer_text`
- `cited_chunk_ids`
- `retrieval_confidence`
- `citation_support_score`
- `completeness_score`
- `final_confidence`
- `raw_prompt`
- `model_name`

### Por qué existe

Porque una respuesta profesional debe ser auditable y evaluable.

---

# 5. Elección de stack: qué usar y por qué

Para este proyecto, la elección pedagógica recomendada es **PostgreSQL + pgvector + Postgres FTS / sparse retrieval local**. Mantiene el core visible, evita coste obligatorio y reduce piezas externas mientras aprendes retrieval, chunking, grounding y evaluación.

## 5.1. PostgreSQL + pgvector

### Por qué usarlo

- unifica metadatos y vectores en el mismo sistema;
- simplifica bastante el desarrollo;
- permite full-text search además de vectores;
- es suficiente para este proyecto;
- ayuda a entender mejor el modelo de datos.

### Implicación

No tendrás que sincronizar:

- una base relacional por un lado,
- y un vector store separado por otro.

### Alternativas

#### Qdrant
Muy bueno para vector search especializado. En este proyecto debe entenderse como alternativa avanzada u optional adapter futuro, no como default de la v1.

#### Chroma
Muy rápido para demos, menos robusto como imagen profesional a largo plazo.

#### Elasticsearch / OpenSearch
Muy potentes para sparse y search en general, pero añaden complejidad operativa.

### Decisión recomendada para aprender

Para este proyecto:

- **PostgreSQL + pgvector + Postgres FTS / sparse retrieval local** es la elección recomendada.

Te obliga a entender bien retrieval sin meter demasiadas piezas externas y mantiene una baseline local-first, reproducible y sin coste obligatorio.

## 5.2. FastAPI

### Por qué

- asincronía nativa;
- ergonomía muy buena;
- documentación OpenAPI automática;
- estándar de facto en APIs Python modernas.

## 5.3. Python 3.11+

### Por qué

- ecosistema enorme;
- librerías de NLP, embeddings, retrieval y evaluación;
- buena productividad;
- excelente para este tipo de sistema.

## 5.4. Streamlit o frontend simple

Para la demo pública, Streamlit puede ser suficiente.

### Ventaja

- rapidez.

### Desventaja

- menos sensación de producto.

Si el objetivo principal es aprender RAG, Streamlit basta. Si quieres pulir portfolio, un frontend React sencillo puede quedar mejor.

---

# 6. Fase 1 — Ingestión documental y normalización

Aquí empieza el sistema real.

## 6.1. Qué significa “ingestar”

No es solo “leer un archivo”. Ingestar implica:

- detectar tipo;
- extraer texto útil;
- preservar metadatos;
- estandarizar formato;
- registrar el origen;
- evitar trabajo duplicado.

## 6.2. Formatos iniciales

Empieza con:

- Markdown
- TXT
- HTML
- PDF

Eso es suficiente para aprender casi todo lo importante.

## 6.3. Reglas de normalización

El objetivo no es dejar el texto “bonito”. Es dejarlo **consistente** para que retrieval y chunking funcionen bien.

### Limpiezas recomendadas

- unificar saltos de línea;
- colapsar espacios múltiples;
- eliminar headers/footers repetidos en PDF cuando sea posible;
- conservar headings;
- preservar numeración útil (`Art. 4`, `§ 2`, `Capítulo I`);
- no mezclar contenido de páginas distintas sin marcarlo.

### Riesgo común

Normalizar de más.

Si destruyes estructura útil, luego el chunking structure-aware pierde valor.

## 6.4. Checksums

Calcula un hash del archivo crudo y, si quieres ser más fino, también del texto normalizado.

### Por qué es importante

- evita reprocesar archivos idénticos;
- reduce coste de embeddings;
- ayuda a detectar cambios reales.

## 6.5. Adaptador legalize-*

Como vas a trabajar con repos `legalize-*`, necesitas un parser específico.

### Qué debe hacer

- leer frontmatter;
- extraer metadatos como:
  - identificador
  - ELI
  - fecha
  - jurisdicción
  - rango normativo
  - estado
- separar encabezado y cuerpo;
- preservar títulos y estructura.

### Por qué esto importa

En corpus legales, el metadato no es decoración. Puede ser crítico para:

- filtrar por jurisdicción;
- citar mejor;
- priorizar documentos;
- responder con más precisión.

## 6.6. Qué debes aprender aquí

Al terminar esta fase debes entender:

- por qué un corpus sucio degrada retrieval;
- por qué la trazabilidad empieza en la ingestión, no en la generación;
- por qué el documento original y el texto procesado deben convivir.

---

# 7. Fase 2 — Chunking: el corazón del RAG

Esta es una de las decisiones más importantes de todo el proyecto.

## 7.1. Qué es un chunk realmente

Un chunk es una unidad de recuperación.

No es simplemente un trozo arbitrario de texto. Es el compromiso entre:

- contexto suficiente;
- granularidad suficiente.

## 7.2. El problema del chunking

### Si el chunk es demasiado pequeño

- pierde contexto;
- genera respuestas fragmentadas;
- obliga a recuperar muchos chunks;
- el embedding puede volverse ambiguo.

### Si el chunk es demasiado grande

- mezcla temas distintos;
- diluye la relevancia;
- consume ventana de contexto;
- empeora precisión del retrieval.

## 7.3. Estrategia 1: Fixed-size con overlap

### Qué es

Divides por número de caracteres o tokens, con cierto solapamiento entre chunks adyacentes.

### Ventajas

- simple;
- robusta;
- baseline universal;
- fácil de comparar.

### Desventajas

- ignora estructura semántica;
- puede cortar una idea a la mitad;
- puede mezclar secciones distintas.

### Cuándo usarla

- como baseline;
- en corpus poco estructurados;
- para comparar con otras estrategias.

## 7.4. Estrategia 2: Structure-aware

### Qué es

Usa headings, secciones o delimitadores lógicos para decidir dónde cortar.

### Ventajas

- preserva significado documental;
- ideal para leyes, contratos, manuales y documentación técnica;
- mejora interpretabilidad del chunk.

### Desventajas

- depende de una estructura bien formada;
- algunos documentos están mal etiquetados;
- los bloques pueden quedar demasiado grandes o demasiado desiguales.

### Cuándo usarla

- casi siempre en corpus legales y docs bien estructurados.

## 7.5. Estrategia 3: Semantic chunking

### Qué es

Intentas detectar cambios de tema semántico usando embeddings o heurísticas sobre frases/párrafos.

### Ventajas

- puede capturar fronteras de tema reales;
- útil en documentos largos con estructura irregular.

### Desventajas

- más compleja;
- más costosa;
- menos determinista;
- más difícil de depurar.

### Recomendación

No la usaría como primera estrategia. La introduciría después de tener buenas baselines.

## 7.6. Qué guardar por cada chunk

Siempre:

- `chunk_index`
- `start_offset`
- `end_offset`
- `heading`
- `page_number` si aplica
- `strategy`
- `document_id`

### Por qué

Sin offsets y metadatos de origen:

- no puedes reconstruir contexto;
- no puedes citar bien;
- no puedes comparar estrategias.

## 7.7. Overlap: por qué existe

El overlap existe para no perder información en fronteras artificiales.

### Ejemplo

Si una definición termina al inicio del chunk siguiente, sin overlap podrías perder una parte esencial.

### Riesgo

Demasiado overlap genera duplicidad y ruido.

### Regla práctica inicial

- 10%–20% del tamaño del chunk.

## 7.8. Qué debes aprender aquí

Al terminar esta fase debes poder explicar:

- por qué chunking no es un detalle menor;
- cómo afecta al retrieval y a la calidad final;
- por qué dos sistemas con el mismo LLM pueden comportarse muy distinto solo por chunking.

---

# 8. Fase 3 — Indexado híbrido: dense + sparse

Aquí pasamos del texto al sistema de búsqueda.

## 8.1. Dense retrieval

### Qué hace

Convierte chunks y preguntas en vectores semánticos.

La similitud vectorial intenta capturar significado, no solo coincidencia literal.

### Ejemplo

Pregunta: “derecho a la intimidad”

Dense retrieval puede recuperar texto que habla de “privacidad” aunque no use exactamente la misma palabra.

### Ventajas

- buena generalización semántica;
- útil para reformulaciones;
- potente cuando la redacción varía.

### Desventajas

- puede fallar con términos exactos;
- puede sobre-semanticar;
- no siempre captura bien nombres de funciones, códigos, artículos o claves de configuración.

## 8.2. Sparse retrieval

### Qué hace

Busca coincidencias léxicas y términos exactos.

En este proyecto, la baseline recomendada es Postgres Full Text Search; también podrías montar BM25 local para comparativas o fallback.

### Ventajas

- excelente para términos exactos;
- muy útil en documentación técnica y legal;
- mejor para códigos, nombres concretos, artículos concretos.

### Desventajas

- peor con reformulaciones semánticas;
- depende más de las palabras exactas.

## 8.3. Por qué híbrido

Porque dense y sparse se complementan.

Dense responde bien a:

- equivalencias de significado;
- preguntas formuladas de otra manera;
- búsquedas conceptuales.

Sparse responde bien a:

- términos exactos;
- referencias normativas;
- siglas;
- nombres propios;
- keywords técnicas.

En corpus profesionales, usar solo uno suele ser peor que combinarlos bien.

## 8.4. Elección de embeddings

Los proveedores configurables son una buena decisión, siempre que el default de aprendizaje siga siendo local-first.

### Por qué

- desacoplas el sistema del proveedor;
- puedes comparar calidad/coste;
- evitas lock-in;
- puedes ejecutar tests con mock generation y evaluación retrieval-only sin depender de APIs externas.

### Qué abstraer

Define una interfaz del tipo:

- `embed_documents(texts)`
- `embed_query(text)`

### Qué aprender aquí

Que embeddings no son “mágicos”: son una representación. Su calidad depende de:

- el modelo;
- el dominio;
- el tamaño del chunk;
- la calidad del texto;
- la pregunta del usuario.

## 8.5. Sincronización de índices

Cada chunk debe existir de forma coherente en:

- tabla de chunks;
- índice vectorial;
- índice sparse / tsvector.

### Riesgo

Si una inserción falla a medias, tendrás inconsistencia.

### Solución

Piensa en el indexado como una operación controlada:

- insert metadata;
- compute embedding;
- persist vector;
- update sparse representation;
- commit.

---

# 9. Fase 4 — Orquestador de retrieval y fusión

Aquí ocurre una parte central del valor del sistema.

## 9.1. Query-time flow

Cuando llega una pregunta:

1. normalizas la pregunta;
2. generas embedding de query;
3. lanzas búsqueda dense;
4. lanzas búsqueda sparse;
5. fusionas ambos rankings;
6. opcionalmente rerankeas;
7. construyes el contexto final.

## 9.2. Reciprocal Rank Fusion (RRF)

RRF combina rankings distintos sin mezclar directamente scores incompatibles.

La fórmula clásica es:

\[
RRF(d) = \sum_i \frac{1}{k + rank_i(d)}
\]

Donde:

- \( d \) es un documento o chunk;
- \( rank_i(d) \) es la posición del chunk en el ranking \( i \);
- \( k \) es una constante de suavizado.

## 9.3. Por qué RRF es tan útil

Porque los scores dense y sparse viven en escalas muy distintas.

- una similitud vectorial no significa lo mismo que un score BM25;
- combinarlos linealmente suele ser frágil;
- RRF usa orden relativo, no score absoluto.

### Ventaja práctica

Es una forma muy robusta de fusionar buscadores heterogéneos.

## 9.4. ¿Hace falta ponderación?

Puedes introducir pesos si quieres, pero no empieces por ahí.

### Recomendación

Primero implementa RRF estándar.

Después, si los datos lo justifican, prueba:

- dense más fuerte;
- sparse más fuerte.

Sin evaluación, tocar pesos es tuning a ciegas.

## 9.5. Reranking

### Qué es

Tomas los mejores N candidatos tras la fusión y los vuelves a ordenar con un modelo más preciso.

### Por qué existe

La búsqueda inicial optimiza recall razonable. El reranker optimiza precisión fina.

### Modelos posibles

- cross-encoder pequeño;
- LLM-as-judge;
- modelo específico de reranking.

### Recomendación pedagógica

Empieza con un reranker opcional. Es muy útil para aprender, pero no debe bloquear el MVP.

### Trade-off

- mejor precisión
- más latencia
- más coste

## 9.6. Qué debes aprender aquí

Al terminar esta fase debes poder explicar:

- por qué retrieval inicial y reranking son etapas distintas;
- por qué dense y sparse no compiten, se complementan;
- por qué RRF es una fusión pragmática muy sólida.

---

# 10. Fase 5 — Construcción del contexto

Esta fase suele estar infravalorada.

## 10.1. El LLM no ve “resultados”, ve texto formateado

Una cosa es cómo recuperas chunks. Otra es cómo se los presentas al modelo.

## 10.2. Qué debe contener cada bloque de contexto

Idealmente cada chunk enviado al LLM debe incluir:

- un identificador numérico (`[1]`, `[2]`...);
- nombre o título del documento;
- metadato útil (jurisdicción, sección, artículo, página);
- texto del chunk.

### Ejemplo conceptual

`[3] Documento: Ley X | Sección: Artículo 4 | Página: 12`

texto del chunk...

## 10.3. Por qué numerar chunks

Porque:

- facilita forzar citas;
- simplifica verificación posterior;
- hace el prompt más claro.

## 10.4. Cuántos chunks mandar

No hay número mágico.

### Pocos chunks

- puedes perder contexto.

### Demasiados chunks

- metes ruido;
- aumentas latencia y coste;
- el modelo se dispersa.

### Regla de partida

- tras reranking, top 4–8 chunks suele ser un punto inicial razonable.

## 10.5. Diversidad del contexto

No siempre quieres los 5 chunks casi idénticos del mismo documento.

En algunos casos conviene:

- limitar chunks repetidos por documento;
- o asegurar cierta diversidad de fuentes.

Esto es especialmente importante en multi-hop o cuando el corpus tiene redundancia.

---

# 11. Fase 6 — Generación grounded y citas

Aquí entramos en la parte que más se ve, pero no la más importante.

## 11.1. Qué significa “grounded”

Que la respuesta debe estar apoyada en el contexto recuperado y no en la imaginación libre del modelo.

## 11.2. Reglas de un buen prompt grounded

El prompt debe dejar claro:

- responde solo con el contexto proporcionado;
- no inventes información;
- si no hay base suficiente, dilo explícitamente;
- cita las fuentes con referencias como `[1]`, `[2]`;
- no cites fragmentos no proporcionados.

## 11.3. Error clásico

Un prompt muy agresivo del tipo “responde siempre” destruye grounding.

Tienes que permitir la salida de incertidumbre.

## 11.4. Qué formato de respuesta conviene

Para este proyecto, una respuesta útil puede incluir:

- respuesta principal;
- citas inline;
- nota de limitación si falta contexto;
- lista opcional de fuentes usadas.

## 11.5. Verificación de citas

Tu borrador decía “programaremos una lógica que valide que el número [1] realmente corresponde a un documento recuperado”. Eso es correcto, pero insuficiente como concepto profesional.

Hay dos niveles de verificación.

### Nivel 1: verificación sintáctica

Comprueba que:

- `[1]` existe;
- `[1]` corresponde a un chunk realmente recuperado.

### Nivel 2: verificación semántica

Comprueba que la afirmación apoyada por `[1]` realmente está soportada por ese chunk.

### Por qué esto importa

Un modelo puede citar un chunk real y aun así usarlo mal.

Ese es uno de los fallos más peligrosos de un RAG con citas.

## 11.6. Manejo del “I don’t know”

Esto es un rasgo de madurez.

### Regla profesional

Si retrieval confidence o soporte documental son insuficientes:

- no improvises;
- devuelve una respuesta estructurada diciendo qué sí se encontró y qué no.

### Qué aprendizaje te llevas

Que una buena respuesta en RAG no es solo “responder mucho”, sino saber **cuándo no debes responder fuerte**.

---

# 12. Fase 7 — Scores de confianza

Esto no tiene que ser perfecto para ser útil.

## 12.1. Qué queremos medir

Un score compuesto puede apoyarse en:

- calidad del retrieval;
- cobertura de citas;
- consistencia de la respuesta;
- completitud respecto a la pregunta.

## 12.2. Retrieval confidence

Heurísticas iniciales:

- fuerza del top result;
- separación entre top resultados y el resto;
- presencia simultánea en dense y sparse;
- acuerdo entre retrievers;
- score del reranker.

## 12.3. Citation coverage

Qué porcentaje de afirmaciones relevantes tiene cita verificable.

## 12.4. Completeness

Si la pregunta tiene varias partes, ¿se han cubierto todas?

## 12.5. Por qué no obsesionarse demasiado aquí al principio

Los confidence scores perfectos son difíciles.

Lo importante al principio es tener una heurística razonable y observable.

---

# 13. Fase 8 — Evaluación: donde un RAG deja de ser una demo

Este es el punto que más separa a un ingeniero serio de alguien que solo montó un tutorial.

## 13.1. Golden dataset

Necesitas un conjunto de preguntas de referencia.

### Tipos de preguntas que debe incluir

- lookup simple;
- pregunta con término exacto;
- pregunta semántica para dense retrieval;
- multi-hop entre varios documentos;
- pregunta ambigua;
- pregunta sin respuesta en el corpus.

## 13.2. Qué anotar por cada pregunta

Idealmente:

- respuesta esperada o criterio de corrección;
- documentos/chunks que deberían aparecer;
- si existe o no respuesta en el corpus.

## 13.3. Métricas importantes

### Hit@k

¿Aparece algún chunk relevante entre los primeros k?

### MRR / ranking usefulness

¿Qué tan arriba aparecen los resultados relevantes?

### Faithfulness

¿La respuesta está apoyada en el contexto?

### Citation accuracy

¿Las citas realmente soportan las afirmaciones?

### Answer correctness

¿La respuesta responde bien a la pregunta?

## 13.4. Comparativas que sí debes hacer

- dense-only vs hybrid;
- chunking fijo vs structure-aware;
- con reranker vs sin reranker.

## 13.5. Qué te enseña esta fase

Que en RAG no se mejora por intuición, sino por medición.

---

# 14. Fase 9 — API y demo

Una vez el motor funciona, hay que exponerlo.

## 14.1. API mínima con FastAPI

Endpoints iniciales:

- `POST /v1/ingest`
- `GET /v1/documents`
- `POST /v1/ask`
- `GET /v1/health`

## 14.2. Qué debe devolver `/v1/ask`

No solo el texto final.

También:

- respuesta;
- citas;
- metadatos de fuentes;
- scores de confianza;
- chunks recuperados opcionalmente;
- modo usado (`dense`, `hybrid`, etc.).

## 14.3. Demo UI

La UI debe ayudarte a aprender, no solo a impresionar.

Muestra:

- la respuesta;
- las citas clicables;
- los chunks recuperados;
- las puntuaciones;
- comparativa hybrid vs dense-only.

Eso te permitirá depurar visualmente el sistema.

---

# 15. Fase 10 — Dockerización y reproducibilidad

Un sistema serio debe poder levantarse de forma consistente.

## 15.1. Qué dockerizar

Como mínimo:

- API
- base de datos Postgres/pgvector
- demo UI si la hay

No incluyas Qdrant por defecto en Docker Compose para la v1. Si aparece más adelante, debe ser un optional adapter documentado.

## 15.2. Por qué esto importa

- facilita pruebas;
- facilita compartir el repo;
- da sensación profesional;
- te obliga a pensar en dependencias reales.

## 15.3. Seed corpus

Incluye un corpus pequeño de ejemplo y un script de indexado.

### Por qué

Porque si alguien clona tu repo y no puede probar nada rápido, el valor del proyecto cae mucho.

---

# 16. Orden realista de implementación

No implementes todo de golpe.

## Secuencia recomendada

### Paso 1
Ingestión básica + modelo `Document`

### Paso 2
Chunking fijo + persistencia de `Chunk`

### Paso 3
Dense retrieval básico con pgvector

### Paso 4
Sparse retrieval básico con Postgres FTS o implementación local

### Paso 5
Hybrid retrieval con RRF

### Paso 6
Prompt grounded + respuesta con citas simples

### Paso 7
UI / API mínima

### Paso 8
Evaluación básica

### Paso 9
Structure-aware chunking

### Paso 10
Reranking

### Paso 11
Citation verification más fuerte

### Paso 12
Confidence scoring

### Paso 13
Docker y polish

### Por qué este orden

Porque así siempre tienes un sistema funcionando y cada mejora se puede medir contra una baseline.

---

# 17. Errores típicos que debes evitar

## 17.1. Obsesionarte con el LLM antes que con retrieval

Error muy común.

La mayor parte del rendimiento real viene de:

- corpus,
- chunking,
- retrieval,
- contexto,

más que del “prompt mágico”.

## 17.2. Meter demasiada abstracción demasiado pronto

Si introduces 20 capas, pierdes visibilidad.

## 17.3. No guardar metadatos suficientes

Esto mata la auditabilidad.

## 17.4. No evaluar con preguntas sin respuesta

Sin esto, no aprendes a manejar incertidumbre.

## 17.5. No comparar estrategias

Si no comparas, no sabes si una mejora es real.

## 17.6. Creer que una cita equivale a soporte real

No siempre. Por eso la verificación semántica importa.

---

# 18. Qué conocimiento profesional deberías haber adquirido al final

Si haces bien este proyecto, deberías salir sabiendo explicar con claridad:

## Arquitectura
- qué componentes forman un RAG serio;
- por qué retrieval y generation son capas distintas;
- cómo se separa dominio de infraestructura.

## Datos
- por qué la calidad del corpus manda;
- cómo normalizar documentos;
- por qué checksums y versionado importan.

## Chunking
- cómo elegir estrategia;
- por qué afecta tanto al rendimiento;
- qué trade-offs existen.

## Retrieval
- dense vs sparse;
- por qué híbrido suele ganar;
- cómo funciona RRF;
- cuándo merece la pena rerankear.

## Grounding
- cómo construir prompts grounded;
- cómo forzar citas;
- por qué el no-answer handling es esencial.

## Evaluación
- qué métricas importan;
- cómo construir un golden dataset;
- cómo comparar configuraciones con datos.

## Ingeniería
- cómo empaquetar el sistema en API;
- cómo dockerizarlo;
- cómo hacerlo reproducible;
- cómo explicar decisiones técnicas en entrevistas.

---

# 19. Alternativas razonables y cuándo elegirlas

## En lugar de Postgres + pgvector

### Qdrant
Elígelo si priorizas una experiencia vectorial más especializada y aceptas añadir una pieza operativa. Para la v1 de este repo queda fuera de la baseline y solo debe aparecer como optional adapter futuro.

### Elasticsearch/OpenSearch
Elígelo si quieres una capa search muy potente y aceptas más complejidad operativa.

## En lugar de full-text en Postgres

### BM25 externo
Puede darte más control o métricas más clásicas, pero añade piezas.

## En lugar de Streamlit

### React frontend
Más trabajo, mejor presentación.

## En lugar de citation verification manual

### LLM-as-judge
Más sofisticado, más coste y más opacidad.

Durante las primeras fases, mock generation y evaluación retrieval-only son perfectamente válidas: permiten aislar retrieval, chunking, grounding y citation verification antes de introducir variabilidad de un LLM real.

La elección correcta depende del objetivo. Aquí tu objetivo principal es **aprender de verdad** sin inflar el sistema más de la cuenta ni prometer compliance legal avanzado.

---

# 20. Propuesta final de roadmap de aprendizaje

## Semana 1
- arquitectura
- ingestión
- normalización
- chunking baseline
- persistencia

## Semana 2
- dense retrieval
- sparse retrieval
- hybrid + RRF
- prompt grounded
- API mínima

## Semana 3
- evaluation dataset
- comparativas
- structure-aware chunking
- reranker opcional

## Semana 4
- citation verification
- confidence score
- UI/demo
- dockerización
- documentación final

---

# 21. Criterio de éxito del proyecto

El proyecto es bueno si al final puedes demostrar cuatro cosas:

1. **funciona**;
2. **puedes explicar por qué funciona**;
3. **puedes medir cuándo falla**;
4. **puedes justificar cada decisión importante con trade-offs reales**.

Si consigues eso, no habrás construido solo un RAG. Habrás aprendido a pensar como alguien que diseña sistemas RAG de nivel profesional.
