# CLAUDE.md

Guide pour Claude Code travaillant sur ce dépôt.

## Vue d'ensemble

RAG (Retrieval-Augmented Generation) pédagogique : indexation de fichiers **texte**
(`.txt`, `.md`) dans **Qdrant**, questions/réponses via **Mistral** (embeddings + chat),
chaque réponse citant sa source (`rapport.txt (l. 12-40)`).

La pièce maîtresse est le **chunking sémantique piloté par LLM** : le modèle décide des
frontières, jamais du contenu. Le texte des chunks est reconstruit par du code
déterministe depuis le document original, puis vérifié.

Le code est **en français** (docstrings, commentaires, messages d'erreur, prompts).
**Conserver cette langue** dans tout ajout ou modification.

Le projet est volontairement dépourvu de reranking, d'observabilité, de cache, de
recherche hybride, d'OCR et de support PDF. Ne pas introduire ces briques sans demande
explicite : la simplicité est un objectif du projet, pas un oubli.

## Architecture

Chaque étage a une classe abstraite (ABC) + une implémentation concrète.
`RAGPipeline` ne dépend que des abstractions et reçoit ses dépendances par injection.

| Fichier | Rôle | ABC | Implémentation |
|---|---|---|---|
| `rag/config.py` | Configuration depuis `.env` | — | `Settings` (dataclass *frozen*) |
| `rag/models.py` | Objets métier entre étages | — | `Block`, `ParsedDocument`, `PlannedChunk`, `ChunkPlan`, `Chunk`, `RetrievedChunk`, `Answer` |
| `rag/parser.py` | 1. Texte → blocs identifiés | `DocumentParser` | `TextParser` |
| `rag/planner.py` | 2. Blocs → plan (identifiants seuls) | `ChunkPlanner`, `LLMChunkPlanner` | `MistralChunkPlanner`, `OllamaChunkPlanner`, `HeuristicChunkPlanner` |
| `rag/validator.py` | 3. Validation stricte du plan | — | `PlanValidator` |
| `rag/reconstructor.py` | 4. Plan → texte des chunks | — | `ChunkReconstructor` |
| `rag/semantic_chunker.py` | Orchestration lots/retry/repli | — | `SemanticChunker` |
| `rag/embedder.py` | 5. Vectorisation | `Embedder` | `MistralEmbedder` |
| `rag/vector_store.py` | 6. Stockage + recherche | `VectorStore` | `QdrantVectorStore` |
| `rag/generator.py` | 7. Prompt + LLM | `AnswerGenerator` | `MistralGenerator` |
| `rag/pipeline.py` | Orchestration générale | — | `RAGPipeline` |
| `rag/utils.py` | `retry()` exponentiel | — | — |

Deux flux :
- **Ingestion** : `.txt` → `Block[]` → `ChunkPlan` → validation → `Chunk[]` → embeddings → Qdrant
- **Interrogation** : question → embedding → `search(top_k)` → prompt numéroté → LLM → `Answer`

Points d'entrée : `app.py` (Streamlit), `main.py` (CLI), `build_subcorpus.py` (extraction
d'un sous-corpus du benchmark), `smoke_test.py` (essai hors-ligne),
`tests/` (53 tests `unittest`, aucun appel réseau).

### Règles d'extension

- Nouveau format de document / autre base vectorielle / autre fournisseur LLM :
  écrire une classe implémentant l'ABC correspondante, l'injecter dans `RAGPipeline`.
  Ne pas modifier `pipeline.py` pour un cas particulier.
- `RAGPipeline.from_settings()` est la seule fabrique qui connaît Mistral + Qdrant.
- Ne pas faire circuler de dictionnaires anonymes entre étages : utiliser les
  dataclasses de `rag/models.py`.

### Invariants du chunking (à ne jamais casser)

1. **Le LLM ne produit que des identifiants.** Aucun champ textuel de sa réponse n'est
   lu (`MistralChunkPlanner._parse` ignore tout sauf `id`, `block_ids`, `section_path`).
2. **Le texte d'un chunk** est `separator.join(bloc.text)` — rien d'autre.
3. **Tout plan passe par `PlanValidator`**, y compris celui du repli déterministe.
4. **Un bloc ne dépasse jamais `MAX_BLOCK_CHARS`** : le parser redécoupe les
   paragraphes trop longs sur des frontières de phrases, offsets exacts conservés.
5. **`block.text == document.text[block.char_start:block.char_end]`** : vérifié par le
   parser (`_audit`) puis par `PlanValidator.verify_chunks`.
6. **`section_path`** ne contient que des textes de titres du document ; le chemin
   proposé par le LLM est recalé sur la pile de titres réellement ouverts
   (`SemanticChunker._repair_section_paths`).

## Commandes

```bash
# Tests (aucune dépendance externe, aucun appel API, pas besoin de Qdrant)
python -m unittest discover -s tests -t .

# Essai de bout en bout sans crédits API (Qdrant ignoré s'il est éteint)
python smoke_test.py

# Base vectorielle
docker compose up -d              # Qdrant sur http://localhost:6343/dashboard

# CLI
python main.py chunks data/test_rapport.txt --blocks   # inspecter le découpage
python main.py ingest data/                            # indexer un dossier
python main.py ingest ../corpus --recursive
python main.py ask "Quel est le budget ?"
python main.py status
python main.py reset

# Sous-corpus de travail (le benchmark complet n'est pas indexable ici)
python build_subcorpus.py --distracteurs 20000   # -> bench/ : 722 gold + distracteurs
python main.py ingest bench

# Interface web
streamlit run app.py
```

Pour valider un changement : `python -m unittest discover -s tests -t .` d'abord,
`python smoke_test.py` ensuite. Les tests couvrent bloc manquant, bloc dupliqué,
mauvais ordre, `block_id` inexistant, texte modifié, hallucination, retry, repli,
réparation du `section_path` et ingestion complète.

## Configuration

Tout passe par `.env` (chargé par `python-dotenv`), lu une seule fois dans
`Settings.from_env()`. Seul `MISTRAL_API_KEY` est obligatoire. Voir `.env.example`.

| Variable | Défaut | Note |
|---|---|---|
| `MISTRAL_API_KEY` | — | obligatoire, sinon `RuntimeError` au démarrage |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `mistral-embed` / `1024` | doit correspondre à la collection |
| `LLM_MODEL` | `mistral-small-latest` | réponses |
| `PLANNER_PROVIDER` | `mistral` | `ollama` = découpage par un modèle local, zéro crédit |
| `PLANNER_MODEL` | *(= `LLM_MODEL` ou `OLLAMA_MODEL`)* | modèle du planner |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `localhost:11434` / `llama3.2:3b` | serveur et modèle local |
| `OLLAMA_NUM_CTX` / `OLLAMA_TIMEOUT` | `8192` / `180` | fenêtre de contexte, secondes par appel |
| `CHUNKING_MODE` | `semantic` | `heuristic` = zéro appel LLM au découpage |
| `TARGET_CHUNK_CHARS` / `MAX_CHUNK_CHARS` | `1200` / `2000` | taille visée / plafond garanti |
| `MIN_CHUNK_CHARS` | `200` | seuil de fusion d'un chunk trop court |
| `MAX_BLOCK_CHARS` | `1200` | doit rester ≤ `MAX_CHUNK_CHARS` |
| `PLANNER_BATCH_BLOCKS` | `40` | blocs par appel LLM |
| `PLANNER_MAX_ATTEMPTS` | `2` | tentatives avant repli |
| `QDRANT_URL` | `http://localhost:6343` | **6343**, pas 6333 (voir ci-dessous) |
| `QDRANT_COLLECTION` | `documents` | |
| `TOP_K` | `4` | |
| `DATA_DIR` | `data` | |

## Pièges connus

- **Port Qdrant 6343** (mappé sur 6333 dans le conteneur) : une autre instance Qdrant
  occupe déjà 6333 sur cette machine. Ne pas « corriger » vers 6333.
- **Version Qdrant figée** (`qdrant/qdrant:v1.19.0`) pour rester alignée sur
  `qdrant-client` : le client refuse un écart de version majeure.
- **Dimension des vecteurs** : changer `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` sans
  recréer la collection lève une `RuntimeError` explicite dans `_ensure_collection()`.
- **Payload Qdrant modifié** : les chunks stockent désormais `line_start`, `line_end`,
  `section_path`, `block_ids` (plus de `page`). Une collection remplie par une version
  antérieure doit être vidée (`python main.py reset`).
- **`ingest_file` supprime d'abord les chunks du document** avant de le réindexer :
  c'est ce qui évite les points orphelins quand un document raccourcit.
- **Reprise d'une ingestion interrompue** : `ingest bench --reprendre` saute les
  documents déjà présents (`vector_store.list_sources()`). C'est sûr parce que les
  chunks d'un document sont toujours écrits en un seul lot : un document indexé l'est
  intégralement. Ne pas lancer `reset` entre-temps, sinon la reprise repart de zéro.
- **Coût du mode `semantic`** : ~1 appel LLM par tranche de 40 blocs. Pour ne rien
  dépenser : `PLANNER_PROVIDER=ollama` (local) ou `CHUNKING_MODE=heuristic`.
- **Le corpus complet est hors de portée** (mesuré sur 150 fichiers tirés au hasard) :
  751 000 appels au planner, 677 M tokens d'entrée, 3,2 M vecteurs (13 Go) et 625 M
  tokens d'embeddings *quel que soit le mode de chunking*, sur une machine à 17 Go.
  Travailler sur le sous-corpus `bench/` (`build_subcorpus.py`), pas sur
  `../EnterpriseRAG-bench` directement.
- **Planner Ollama** : `LLMChunkPlanner` mutualise prompt/schéma/lecture ; seul
  `_complete()` diffère (`/api/chat` via `urllib`, champ `format` = `CHUNK_PLAN_SCHEMA`).
  Ollama éteint → `PlannerError` → repli déterministe, l'ingestion aboutit quand même.
  Un petit modèle (3B) a tendance à replanifier les blocs de contexte : c'est pour ça
  que le prompt utilisateur réénumère explicitement les `block_id` à couvrir.
- **Appels API** : toujours passer par `retry()` de `rag/utils.py` (4 tentatives,
  délai exponentiel) pour absorber les 429 / erreurs réseau.
- **Deux prompts** : `rag/planner.py` (`SYSTEM_PROMPT`, découpage) et
  `rag/generator.py` (`SYSTEM_PROMPT`, réponse citée). Ce sont les premiers leviers
  de qualité.
- **Détection des titres implicites** : une ligne courte isolée finissant par `:` est
  traitée comme un titre. Un faux positif dégrade seulement le `section_path`, jamais
  le contenu — ne pas durcir la règle sans mesurer.

## État actuel du dépôt

- **Dépendances** : installées dans `../.venv` (Python 3.11) — pas de `.venv` dans le
  dossier du projet ; le `python3` du système est en 3.9.6 et n'a que `python-dotenv`.
  Utiliser `../.venv/bin/python`.
- **`.env`** présent avec une clé Mistral valide (le planner LLM a été testé en réel).
- **Qdrant éteint** : le démon Docker ne tourne pas sur cette machine actuellement.
  Les tests et `smoke_test.py` fonctionnent sans lui ; l'ingestion réelle non.
- **`data/`** contient encore d'anciens PDF (`test_rapport.pdf`, etc.) : ils sont
  simplement ignorés, le pipeline ne traite plus que `.txt`/`.md`. Le document de
  démonstration est `data/test_rapport.txt`.
- `.gitignore` couvre `EnterpriseRAG-bench/` et `.env` ; `rag/__pycache__/*.pyc` et
  `.DS_Store` restent versionnés à tort.
