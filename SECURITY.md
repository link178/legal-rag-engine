# Security Policy

## Supported versions

This is an early-stage open-source project. Security fixes are applied on a best-effort basis on the default branch.

## Reporting a vulnerability

Please **do not** open a public issue for security-sensitive reports.

- Open a [GitHub Security Advisory](https://docs.github.com/code-security/security-advisories/about-github-security-advisories) (private) if you have access, or
- Contact the repository maintainer through a private channel.

## General guidance

- **Do not commit secrets**, API keys, or production database credentials. Use `.env` locally (never committed; see `.gitignore`).
- The **default `DATABASE_URL` in `.env.example` matches the local Docker Compose Postgres** (`legal_rag` user/password). Treat those values as **development-only**; do not reuse them in production.

This project’s baseline is **local-first** and does not require paid external services for default tests or the mock-generation demo.
