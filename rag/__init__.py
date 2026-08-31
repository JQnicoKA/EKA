"""Package RAG : un pipeline Retrieval-Augmented Generation minimaliste.

Chaîne complète : PDF -> chunks -> embeddings -> Qdrant -> recherche -> LLM Mistral.
"""

from .config import Settings
from .models import Answer, Chunk, Page, RetrievedChunk
from .pipeline import RAGPipeline

__all__ = ["Settings", "RAGPipeline", "Answer", "Chunk", "Page", "RetrievedChunk"]
