"""Document de test partagé par les cas de test."""

from __future__ import annotations

from rag.parser import TextParser

DOCUMENT = """Architecture de la plateforme

## Backend
Le backend utilise Python 3.11 et FastAPI.
Les traitements longs passent par une file Redis.

- Service d'authentification
- Service de facturation

## Frontend
L'interface est écrite en TypeScript avec React.

---

## Déploiement
Le déploiement est piloté par Terraform.
"""


def parsed(text: str = DOCUMENT, source: str = "architecture.txt", max_block_chars: int = 1200):
    return TextParser(max_block_chars=max_block_chars).parse_text(text, source=source)
