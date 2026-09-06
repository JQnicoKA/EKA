# CLAUDE.md

Guide pour Claude Code travaillant sur ce dépôt.

## Vue d'ensemble

RAG (Retrieval-Augmented Generation) minimaliste et pédagogique : indexation de
fichiers **`.txt`** dans **Qdrant**, questions/réponses via **Mistral** (embeddings +
chat), chaque réponse citant sa source (`notes.txt (l. 12-40)`).

Le code est **en français** (docstrings, commentaires, messages d'erreur, prompts).
**Conserver cette langue** dans tout ajout ou modification.

Le projet est volontairement dépourvu de reranking, d'observabilité, de cache, de
recherche hybride. Ne pas introduire ces briques sans demande explicite : la simplicité
est un objectif du projet, pas un oubli.

## Architecture

Cinq étages, chacun avec une classe abstraite (ABC) + une implémentation concrète.
`RAGPipeline` ne dépend que des abstractions et reçoit ses dépendances par injection
dans son constructeur.

| Fichier | Rôle | ABC | Implémentation |
|---|---|---|---|
| `rag/config.py` | Configuration depuis `.env` | — | `Settings` (dataclass *frozen*) |
| `rag/models.py` | Objets métier entre étages | — | `Document`, `Chunk`, `RetrievedChunk`, `Answer` |
| `rag/loader.py` | 1. Lecture des `.txt` | `DocumentLoader` | `TextLoader` |
| `rag/chunker.py` | 2. Découpage | — | `TextChunker` |
| `rag/embedder.py` | 3. Vectorisation | `Embedder` | `MistralEmbedder` |
| `rag/vector_store.py` | 4. Stockage + recherche | `VectorStore` | `QdrantVectorStore` |
| `rag/generator.py` | 5. Prompt + LLM | `AnswerGenerator` | `MistralGenerator` |
| `rag/pipeline.py` | Orchestration | — | `RAGPipeline` |
| `rag/utils.py` | `retry()` exponentiel | — | — |

Deux flux :
- **Ingestion** : `.txt` → `Document` → `Chunk[]` → embeddings → upsert Qdrant
- **Interrogation** : question → embedding → `search(top_k)` → prompt numéroté → LLM → `Answer`

Points d'entrée : `app.py` (Streamlit), `main.py` (CLI), `smoke_test.py` (test hors-ligne).

### Règles d'extension

- Nouveau format de document / autre base vectorielle / autre fournisseur LLM :
  écrire une classe implémentant l'ABC correspondante, l'injecter dans `RAGPipeline`.
  Ne pas modifier `pipeline.py` pour un cas particulier.
- `RAGPipeline.from_settings()` est la seule fabrique qui connaît Mistral + Qdrant.
- Ne pas faire circuler de dictionnaires anonymes entre étages : utiliser les
  dataclasses de `rag/models.py`.

## Commandes

```bash
# Base vectorielle (requise pour tout, sauf lecture de code)
docker compose up -d              # Qdrant sur http://localhost:6343/dashboard

# Vérifier le pipeline SANS consommer de crédits API (embedder + LLM factices)
python smoke_test.py

# CLI
python main.py ingest data/                 # indexe tous les .txt d'un dossier
python main.py ingest data/ -r              # ... y compris les sous-dossiers
python main.py ingest data/notes.txt        # un seul fichier
python main.py ask "Quel est le budget ?"
python main.py status                       # documents + nombre de chunks
python main.py reset                        # vide la collection

# Interface web
streamlit run app.py
```

`smoke_test.py` est le moyen privilégié de valider un changement : il rejoue
ingestion → idempotence → recherche → prompt → suppression sur `data/test_rapport.txt`,
dans une collection `smoke_test` séparée, sans appeler Mistral. Il n'y a pas de suite
de tests unitaires.

## Configuration

Tout passe par `.env` (chargé par `python-dotenv`), lu une seule fois dans
`Settings.from_env()`. Seul `MISTRAL_API_KEY` est obligatoire.

| Variable | Défaut | Note |
|---|---|---|
| `MISTRAL_API_KEY` | — | obligatoire, sinon `RuntimeError` au démarrage |
| `EMBEDDING_MODEL` | `mistral-embed` | |
| `EMBEDDING_DIMENSION` | `1024` | doit correspondre à la collection Qdrant |
| `LLM_MODEL` | `mistral-small-latest` | |
| `QDRANT_URL` | `http://localhost:6343` | **6343**, pas 6333 (voir ci-dessous) |
| `QDRANT_COLLECTION` | `documents` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | overlap < size, validé au démarrage |
| `TOP_K` | `4` | |
| `DATA_DIR` | `data` | |

## Pièges connus

- **Port Qdrant 6343** (mappé sur 6333 dans le conteneur) : une autre instance Qdrant
  occupe déjà 6333 sur cette machine. Ne pas « corriger » vers 6333.
- **Version Qdrant figée** (`qdrant/qdrant:v1.19.0`) pour rester alignée sur
  `qdrant-client` : le client refuse un écart de version majeure.
- **Dimension des vecteurs** : changer `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` sans
  recréer la collection lève une `RuntimeError` explicite dans `_ensure_collection()`.
- **IDs déterministes** : `uuid5(namespace, "source:index")`. Réindexer le même
  fichier met à jour les points au lieu de dupliquer — mais s'il produit moins de
  chunks qu'avant, les points en surplus subsistent (`delete_source()` d'abord).
- **Texte lu sans nettoyage** dans `TextLoader` : c'est la condition pour que
  `line_start`/`line_end` correspondent au fichier réel. Ne pas y ajouter de
  normalisation d'espaces ou de sauts de ligne.
- **Format unique** : `.txt` seulement (`TextLoader.SUFFIXES`). Un autre format =
  une nouvelle classe implémentant `DocumentLoader`, pas une condition dans le loader.
- **Appels API** : toujours passer par `retry()` de `rag/utils.py` (4 tentatives,
  délai exponentiel) pour absorber les 429 / erreurs réseau.
- **Prompt système** : dans `rag/generator.py` (`SYSTEM_PROMPT`). C'est le premier
  levier de qualité ; il impose de répondre uniquement à partir des extraits et de
  citer `[1]`, `[2]`.

## État actuel du dépôt

Environnement fonctionnel : `.venv` (Python 3.11) à la racine du dossier parent,
Qdrant démarré sur le port 6343, `.env` renseigné avec `MISTRAL_API_KEY`.
`python smoke_test.py` passe de bout en bout.

Le corpus de benchmark `EnterpriseRAG-bench/` (511 962 fichiers `.txt`) est posé à
côté du dépôt et ignoré par git. Le bouton « Indexer EnterpriseRAG-bench » de
`app.py` en indexe un sous-ensemble borné : l'ingestion fait **un appel d'embeddings
par document**, indexer le corpus entier tel quel n'est pas réaliste.

La branche `semantic-chunking` conserve une version à découpage sémantique piloté par
LLM (parser en blocs, planner, validateur, reconstructeur), mise de côté car trop
coûteuse en appels API pour le bénéfice constaté.
