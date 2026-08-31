# CLAUDE.md

Guide pour Claude Code travaillant sur ce dépôt.

## Vue d'ensemble

RAG (Retrieval-Augmented Generation) minimaliste et pédagogique : indexation de PDF
dans **Qdrant**, questions/réponses via **Mistral** (embeddings + chat), chaque réponse
citant sa source (`document.pdf (p. 3)`).

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
| `rag/models.py` | Objets métier entre étages | — | `Page`, `Chunk`, `RetrievedChunk`, `Answer` |
| `rag/loader.py` | 1. Extraction texte PDF | `DocumentLoader` | `PDFLoader` (pypdf) |
| `rag/chunker.py` | 2. Découpage | — | `TextChunker` |
| `rag/embedder.py` | 3. Vectorisation | `Embedder` | `MistralEmbedder` |
| `rag/vector_store.py` | 4. Stockage + recherche | `VectorStore` | `QdrantVectorStore` |
| `rag/generator.py` | 5. Prompt + LLM | `AnswerGenerator` | `MistralGenerator` |
| `rag/pipeline.py` | Orchestration | — | `RAGPipeline` |
| `rag/utils.py` | `retry()` exponentiel | — | — |

Deux flux :
- **Ingestion** : PDF → `Page[]` → `Chunk[]` → embeddings → upsert Qdrant
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
python main.py ingest data/                 # indexe tous les PDF d'un dossier
python main.py ingest data/rapport.pdf      # un seul fichier
python main.py ask "Quel est le budget ?"
python main.py status                       # documents + nombre de chunks
python main.py reset                        # vide la collection

# Interface web
streamlit run app.py
```

`smoke_test.py` est le moyen privilégié de valider un changement : il rejoue
ingestion → idempotence → recherche → prompt → suppression sur `data/test_rapport.pdf`,
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
- **IDs déterministes** : `uuid5(namespace, "source:page:index")`. Réindexer le même
  PDF met à jour les points au lieu de dupliquer — mais si le PDF change et produit
  moins de chunks, les anciens points en surplus subsistent (`delete_source()` d'abord).
- **Un chunk n'enjambe jamais deux pages** : c'est ce qui garantit une citation de page
  exacte. Ne pas casser cette propriété dans `TextChunker`.
- **PDF scannés** : `pypdf` n'extrait que le texte natif ; un PDF-image lève une
  `ValueError` explicite. Pas d'OCR (hors périmètre).
- **Appels API** : toujours passer par `retry()` de `rag/utils.py` (4 tentatives,
  délai exponentiel) pour absorber les 429 / erreurs réseau.
- **Prompt système** : dans `rag/generator.py` (`SYSTEM_PROMPT`). C'est le premier
  levier de qualité ; il impose de répondre uniquement à partir des extraits et de
  citer `[1]`, `[2]`.

## État actuel du dépôt

**Rien ne s'exécute en l'état.** Trois blocages, dans l'ordre où ils se déclenchent :

1. **Dépendances absentes** : `mistralai`, `qdrant_client`, `streamlit`, `pypdf` ne sont
   pas installés (pas de `.venv` dans le dossier ; le `python3` du système, en 3.9.6,
   n'a que `python-dotenv`). `app.py:17` / `main.py:18` font `from rag import ...`, qui
   charge `rag/embedder.py` → `import mistralai` : l'`ImportError` remonte *avant* le
   `try/except` de `main.py`, donc trace brute et non message soigné. Concerne aussi
   `smoke_test.py`.
2. **Pas de clé API** : ni `.env`, ni `.env.example` (le README et le message d'erreur y
   renvoient pourtant), ni `MISTRAL_API_KEY` dans l'environnement →
   `Settings.from_env()` lève une `RuntimeError` (`rag/config.py:65`).
3. **Qdrant injoignable** sur `localhost:6343` → `RuntimeError` dans
   `QdrantVectorStore._ensure_collection()` (`rag/vector_store.py:74`).

Les points 2 et 3 sont correctement rattrapés (« Erreur de démarrage » en CLI,
`st.error` avec checklist en Streamlit). `smoke_test.py` ne contourne que l'appel à
Mistral — il lui faut quand même les dépendances et un Qdrant qui répond.

Le code compile sous Python 3.9 (`from __future__ import annotations` partout), même si
le README demande 3.10+.
- `.gitignore` ne contient que `EnterpriseRAG-bench/` : `rag/__pycache__/*.pyc` et
  `.DS_Store` sont versionnés à tort, et `.env` n'y est **pas** ignoré — attention à
  ne jamais committer de clé API.
- `app.py` contient un bloc de travail en cours autour du bouton « Cliquezzz »
  (ingestion en masse d'un dossier `./EnterpriseRAG-bench`). Ce bloc est cassé :
  `with monbouton:` sur un booléen, `Path.getbuffer()` qui n'existe pas, `progress`
  hors boucle, variable `uploaded` hors portée, et le pipeline n'accepte que des PDF
  alors que le code cherche des `*.txt`. Le dossier visé n'est pas présent.
