# RAG minimaliste — Mistral + Qdrant

Un pipeline **Retrieval-Augmented Generation** réduit à l'essentiel : on indexe des
des fichiers texte dans une base vectorielle, puis on interroge un LLM en lui fournissant
uniquement les passages pertinents. Chaque réponse est accompagnée de sa source
(document + lignes).

Pas d'observabilité, pas de reranking, pas de stratégie avancée : l'objectif est
de comprendre le mécanisme de bout en bout.

---

## Le pipeline en deux flux

```
INGESTION (une fois par document)
   .txt ──▶ texte ──▶ chunks ──▶ embeddings ──▶ Qdrant
        loader    chunker    MistralEmbedder   vector_store

INTERROGATION (à chaque question)
   question ──▶ embedding ──▶ recherche top-k ──▶ prompt ──▶ LLM ──▶ réponse + sources
             MistralEmbedder    vector_store    generator   Mistral
```

## Architecture du code

| Fichier | Rôle | Classe principale |
|---|---|---|
| `rag/config.py` | Configuration lue depuis `.env` | `Settings` |
| `rag/models.py` | Objets métier échangés entre étages | `Page`, `Chunk`, `RetrievedChunk`, `Answer` |
| `rag/loader.py` | 1. Lecture des fichiers `.txt` | `TextLoader` |
| `rag/chunker.py` | 2. Découpage en morceaux | `TextChunker` |
| `rag/embedder.py` | 3. Vectorisation | `MistralEmbedder` |
| `rag/vector_store.py` | 4. Stockage + recherche | `QdrantVectorStore` |
| `rag/generator.py` | 5. Prompt + appel au LLM | `MistralGenerator` |
| `rag/pipeline.py` | Orchestration des cinq étages | `RAGPipeline` |
| `app.py` | Interface web (Streamlit) | — |
| `main.py` | Interface en ligne de commande | — |
| `smoke_test.py` | Test du pipeline sans appeler l'API Mistral | — |

Chaque étage possède une **classe abstraite** (`DocumentLoader`, `Embedder`,
`VectorStore`, `AnswerGenerator`) et une implémentation concrète. Le pipeline ne
dépend que des abstractions : remplacer Qdrant par autre chose, ou Mistral par un
autre fournisseur, revient à écrire une nouvelle classe sans toucher au reste.

---

## Installation

**Prérequis** : Python 3.10+, Docker, une clé API Mistral
([console.mistral.ai](https://console.mistral.ai/)).

```bash
# 1. Environnement virtuel + dépendances
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Clé API
cp .env.example .env
# puis éditez .env pour renseigner MISTRAL_API_KEY

# 3. Base vectorielle
docker compose up -d
```

> **Déjà fait sur cette machine** : `.venv` est créé avec les dépendances
> installées, et le conteneur Qdrant tourne. **Il ne reste qu'à créer le `.env`
> et y mettre votre clé Mistral.**

> **Port Qdrant** : ce projet expose Qdrant sur le port **6343**, pas le 6333
> standard, car une autre instance Qdrant occupe déjà le 6333 sur cette machine.
> Les deux bases sont donc totalement indépendantes. Tableau de bord :
> <http://localhost:6343/dashboard>

### Vérifier l'installation sans consommer de crédits API

`smoke_test.py` rejoue tout le pipeline (.txt → chunks → Qdrant → recherche →
construction du prompt) en remplaçant Mistral par de faux composants. Utile pour
vérifier que Qdrant répond avant même d'avoir une clé API :

```bash
python smoke_test.py
```

Un fichier d'exemple (`data/test_rapport.txt`) est fourni pour ces essais.

## Utilisation

### Interface web (recommandé)

```bash
streamlit run app.py
```

Puis, dans le navigateur : importez un `.txt` dans la colonne de gauche, cliquez sur
**Indexer**, et posez votre question.

### Ligne de commande

```bash
python main.py ingest data/               # indexe tous les .txt du dossier
python main.py ingest data/ -r            # ... y compris les sous-dossiers
python main.py ask "Quel est le budget ?" # pose une question
python main.py status                     # nombre de documents / chunks
python main.py reset                      # vide la base
```

---

## Réglages utiles (`.env`)

| Variable | Défaut | Effet |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Taille d'un chunk en caractères. Plus petit = recherche plus précise, mais contexte plus fragmenté. |
| `CHUNK_OVERLAP` | `150` | Recouvrement entre deux chunks. Évite de couper une idée en deux. |
| `TOP_K` | `4` | Nombre de passages envoyés au LLM. Plus élevé = plus de contexte, mais plus de bruit et de coût. |
| `LLM_MODEL` | `mistral-small-latest` | `mistral-large-latest` pour des réponses de meilleure qualité. |

Le prompt système (la consigne donnée au LLM) se trouve dans
`rag/generator.py` — c'est le premier levier à ajuster pour améliorer les réponses.

---

## Détails d'implémentation à connaître

- **Références en numéros de ligne** : un fichier texte n'a pas de pages. Chaque
  chunk retient les lignes d'où il provient, d'où des citations du type
  `notes.txt (l. 12-40)`. Le contenu est lu **sans nettoyage**, faute de quoi les
  numéros ne correspondraient plus au fichier ouvert dans un éditeur.
- **Identifiants déterministes** : l'ID d'un point Qdrant est dérivé de
  `source:index`. Réindexer le même fichier met à jour les points au lieu de
  créer des doublons — mais s'il a raccourci, les points en trop subsistent.
- **Similarité cosinus** : le score affiché va de 0 à 1 ; au-delà de ~0,75 le
  passage est en général très pertinent.
- **Format unique** : seul le `.txt` est accepté (`TextLoader.SUFFIXES`). Ajouter
  le `.md` tient en une entrée de ce tuple ; un autre format demande une nouvelle
  classe implémentant `DocumentLoader`.
- **Réessais automatiques** : les appels API sont retentés 4 fois avec un délai
  exponentiel (`rag/utils.py`), pour absorber les erreurs 429 / réseau.

## Pistes pour la suite

Reranking · recherche hybride (dense + BM25) · découpage sémantique ·
historique de conversation · évaluation (RAGAS) · observabilité · cache d'embeddings.
