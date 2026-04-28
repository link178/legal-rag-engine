# Qdrant (adapter opcional)

**Qdrant no es una dependencia obligatoria de v1** del Legal RAG Engine.

Este directorio queda **reservado** para un posible adapter futuro si se desea almacenamiento vectorial externo además de PostgreSQL/pgvector.

**No introducir** Qdrant todavía en:

- La composición Docker principal del proyecto
- La configuración por defecto del motor
- Las dependencias base del scaffold

Cuando exista implementación, deberá ser conmutable y opcional frente al almacenamiento en Postgres.
