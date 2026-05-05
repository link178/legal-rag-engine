"""HTML loader (.html, .htm) — visible text, stdlib html.parser only."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Comment
from bs4.element import Tag

from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.base import LoadedDocument

_BLOCK_TAGS: frozenset[str] = frozenset(
    {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "td", "th", "tr", "div"}
)


def _strip_comments(soup: BeautifulSoup) -> None:
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()


def _remove_noise(soup: BeautifulSoup) -> None:
    for name in ("script", "style", "noscript"):
        for el in soup.find_all(name):
            el.decompose()
    _strip_comments(soup)


def _collect_visible_blocks(soup: BeautifulSoup) -> list[str]:
    blocks: list[str] = []
    for tag in soup.find_all(_BLOCK_TAGS):
        if not isinstance(tag, Tag):
            continue
        if tag.find_parent(_BLOCK_TAGS):
            continue
        text = tag.get_text(" ", strip=True)
        if text:
            blocks.append(text)
    return blocks


class HtmlLoader:
    supported_extensions: tuple[str, ...] = (".html", ".htm")

    def load(self, path: Path) -> LoadedDocument:
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise DocumentLoadError(f"Failed to read file: {e}") from e

        try:
            soup = BeautifulSoup(raw_bytes, "html.parser")
        except Exception as e:
            raise DocumentLoadError(f"Failed to parse HTML: {e}") from e

        _remove_noise(soup)

        html_title: str | None = None
        if soup.title:
            t = soup.title.get_text(strip=True)
            html_title = t if t else None

        h1_el = soup.find("h1")
        h1_text: str | None = None
        if h1_el and isinstance(h1_el, Tag):
            h = h1_el.get_text(strip=True)
            h1_text = h if h else None

        stem = path.stem
        title = html_title or h1_text or (stem if stem else None)

        blocks = _collect_visible_blocks(soup)
        raw_text = "\n\n".join(blocks)

        if not raw_text.strip():
            raise DocumentLoadError("HTML contains no extractable text")

        meta: dict = {"format": "html"}
        if html_title:
            meta["html_title"] = html_title
        if h1_text:
            meta["h1"] = h1_text

        return LoadedDocument(
            source_path=str(path.resolve()),
            source_type="html",
            raw_text=raw_text,
            title=title,
            metadata=meta,
        )
