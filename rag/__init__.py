"""Package RAG : un pipeline Retrieval-Augmented Generation minimaliste.

Chaîne complète : .txt -> chunks -> embeddings -> Qdrant -> recherche -> LLM Mistral.
"""

from .config import Settings
from .models import Answer, Chunk, Document, RetrievedChunk
from .pipeline import RAGPipeline

__all__ = ["Settings", "RAGPipeline", "Answer", "Chunk", "Document", "RetrievedChunk"]
