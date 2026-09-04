"""Petits utilitaires transverses."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry(
    func: Callable[[], T],
    *,
    attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    description: str = "appel API",
) -> T:
    """Réessaie `func` en cas d'exception, avec un délai exponentiel plafonné.

    Utile face aux erreurs transitoires des API distantes (429 rate limit,
    coupure réseau, 5xx). Après `attempts` échecs, l'exception est propagée.

    Les valeurs par défaut couvrent ~1 minute d'attente cumulée (1+2+4+8+16 s) :
    une limitation de débit soutenue est absorbée plutôt que de faire perdre un
    lot entier de documents.
    """
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:  # noqa: BLE001 - on veut vraiment tout rattraper ici
            last_error = error
            if attempt == attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "Échec de %s (tentative %d/%d) : %s — nouvelle tentative dans %.1fs",
                description, attempt, attempts, error, delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Échec de {description} après {attempts} tentatives.") from last_error
