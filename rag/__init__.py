"""Package RAG : un pipeline Retrieval-Augmented Generation minimaliste.

Chaîne complète :
    .txt -> blocs -> plan de chunking (LLM) -> validation -> chunks
         -> embeddings -> Qdrant -> recherche -> LLM Mistral -> réponse citée.
"""

from .config import Settings
from .models import Answer, Block, Chunk, ChunkPlan, ParsedDocument, PlannedChunk, RetrievedChunk
from .pipeline import RAGPipeline

__all__ = [
    "Settings",
    "RAGPipeline",
    "Answer",
    "Block",
    "Chunk",
    "ChunkPlan",
    "ParsedDocument",
    "PlannedChunk",
    "RetrievedChunk",
]
