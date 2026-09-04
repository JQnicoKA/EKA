"""Objets métier échangés entre les composants du RAG.

Ces dataclasses constituent le "contrat" entre les étages du pipeline :
parser -> planner -> validator -> reconstructor -> embedder -> vector store -> generator.
Elles évitent de faire circuler des dictionnaires anonymes.

Trois familles d'objets :
  - le **document parsé** (`Block`, `ParsedDocument`) : la vérité du document source ;
  - le **plan de découpage** (`PlannedChunk`, `ChunkPlan`) : une simple décision de
    regroupement, produite par un LLM ou une heuristique, qui ne contient
    **jamais** de texte ;
  - les **chunks finaux** (`Chunk`), dont le texte est toujours reconstruit
    depuis le document original.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# 1. Document source
# ----------------------------------------------------------------------
class BlockType:
    """Types de blocs reconnus par le parser (simples constantes, pas un Enum)."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    SEPARATOR = "separator"  # ligne de séparation "---" : non indexable

    #: seuls ces types comptent dans la couverture exigée par le validateur
    INDEXABLE = (HEADING, PARAGRAPH, LIST, TABLE, CODE)


@dataclass(frozen=True)
class Block:
    """Unité atomique du document, identifiée et traçable.

    `char_start` / `char_end` sont des offsets dans `ParsedDocument.text` :
    ils permettent de prouver à tout moment que `text` provient bien du document
    original, et non d'un LLM.
    """

    block_id: str      # identifiant stable dans le document, ex. "B001"
    type: str          # une valeur de BlockType
    text: str          # exactement document.text[char_start:char_end]
    position: int      # rang du bloc dans le document, à partir de 1
    source: str        # nom du fichier d'origine
    char_start: int
    char_end: int
    line_start: int    # première ligne couverte, à partir de 1
    line_end: int      # dernière ligne couverte (incluse)
    level: int = 0     # profondeur du titre (1 = titre principal) ; 0 si non-titre

    @property
    def indexable(self) -> bool:
        """Un bloc non indexable (ligne de séparation) est exclu de la couverture."""
        return self.type in BlockType.INDEXABLE

    @property
    def is_heading(self) -> bool:
        return self.type == BlockType.HEADING


@dataclass(frozen=True)
class ParsedDocument:
    """Résultat du parsing : les blocs du document et le texte de référence."""

    source: str
    text: str            # texte normalisé du document : la seule source de vérité
    blocks: list[Block]

    @property
    def indexable_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.indexable]

    def by_id(self) -> dict[str, Block]:
        return {block.block_id: block for block in self.blocks}


# ----------------------------------------------------------------------
# 2. Plan de découpage (sortie du planner : aucune donnée textuelle)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedChunk:
    """Décision de regroupement : « ces blocs forment un chunk ».

    `section_path` contient des **block_id de titres** du document (et non du
    texte libre) : c'est ce qui rend la hiérarchie vérifiable et non hallucinable.
    """

    id: str
    block_ids: list[str]
    section_path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkPlan:
    """Plan complet pour un document (ou un lot de blocs)."""

    chunks: list[PlannedChunk]

    @property
    def all_block_ids(self) -> list[str]:
        ids: list[str] = []
        for chunk in self.chunks:
            ids.extend(chunk.block_ids)
        return ids


# ----------------------------------------------------------------------
# 3. Chunks finaux
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """Un morceau de texte prêt à être vectorisé et stocké.

    `text` est **toujours** reconstruit à partir des blocs sources ;
    `block_ids` et les lignes assurent la traçabilité chunk -> document.
    """

    source: str
    index: int                             # position du chunk dans le document (0, 1, 2...)
    text: str
    chunk_id: str = ""                     # ex. "C001"
    line_start: int = 0
    line_end: int = 0
    section_path: tuple[str, ...] = ()     # titres successifs, ex. ("Architecture", "Backend")
    block_ids: tuple[str, ...] = ()        # blocs sources, dans l'ordre du document

    @property
    def reference(self) -> str:
        """Référence lisible affichée à l'utilisateur, ex. "notes.txt (l. 12-40)"."""
        if not self.line_start:
            return self.source
        if self.line_end and self.line_end != self.line_start:
            return f"{self.source} (l. {self.line_start}-{self.line_end})"
        return f"{self.source} (l. {self.line_start})"

    @property
    def section(self) -> str:
        """Hiérarchie lisible, ex. "Architecture > Backend"."""
        return " > ".join(self.section_path)


@dataclass(frozen=True)
class RetrievedChunk:
    """Un chunk retrouvé dans la base vectorielle, accompagné de son score."""

    chunk: Chunk
    score: float  # similarité cosinus renvoyée par Qdrant (1.0 = identique)


@dataclass(frozen=True)
class Answer:
    """Réponse finale renvoyée à l'utilisateur."""

    question: str
    text: str                                 # réponse générée par le LLM
    sources: list[RetrievedChunk] = field(default_factory=list)  # passages utilisés
