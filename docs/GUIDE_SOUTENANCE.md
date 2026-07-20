# Guide de compréhension du projet — préparation soutenance

## 1. Vue d'ensemble : qu'est-ce que le projet fait ?

L'application est un **assistant documentaire RAG** (*Retrieval-Augmented
Generation*, « génération augmentée par récupération »). L'utilisateur pose
une question en langage naturel ; l'application retrouve les passages du
corpus les plus proches de la question, puis demande à un LLM de rédiger
une réponse **en s'appuyant uniquement sur ces passages**.

Pourquoi RAG plutôt que d'interroger le LLM directement ?

- Un LLM seul répond depuis sa mémoire d'entraînement : il peut se tromper,
  inventer (**halluciner**), et ne connaît pas nos documents privés.
- Le RAG « ancre » la réponse dans un corpus contrôlé : la réponse est
  vérifiable (on affiche les passages utilisés) et l'application peut dire
  « je ne sais pas » quand l'information n'existe pas dans le corpus.

### Le flux complet d'une question

```
1. Utilisateur tape une question dans Streamlit (app.py)
2. La question est convertie en vecteur (embedding)          [vectorstore.py]
3. ChromaDB compare ce vecteur aux vecteurs des segments
   du corpus et retourne les top-k plus proches               [vectorstore.py]
4. Question + passages sont insérés dans un prompt            [llm.py]
5. Le prompt est envoyé à Ollama (API HTTP locale)            [llm.py]
6. Le LLM (qwen2.5:1.5b) génère la réponse                    [Ollama]
7. Réponse + passages + métriques affichés                    [app.py]
8. La requête est journalisée (suivi MLOps)                   [monitoring.py]
```

L'**indexation** (une seule fois, au démarrage) : le corpus est découpé en
segments, chaque segment est converti en vecteur, les vecteurs sont stockés
dans ChromaDB. C'est la phase « offline » ; les étapes 1-8 sont la phase
« online ».

---

## 2. Rôle de chaque fichier

### `app.py` — interface Streamlit

- Page web : titre, description, formulaire de question, bouton, affichage
  réponse / passages / métriques, historique de session.
- `@st.cache_resource` sur `get_pipeline()` : Streamlit ré-exécute tout le
  script à chaque interaction ; ce décorateur **met en cache** le pipeline
  (donc l'index vectoriel) pour ne pas réindexer le corpus à chaque clic.
- `st.session_state.history` : mémoire de la session navigateur, utilisée
  pour le tableau d'historique des requêtes.
- Gestion d'erreurs : question vide → avertissement ; Ollama coupé →
  message d'erreur clair ; info absente → bandeau d'information.
- Bloc `st.secrets` en tête : sur Streamlit Community Cloud, recopie les
  secrets (ex. `OLLAMA_URL`) dans les variables d'environnement avant le
  chargement de la configuration.

### `rag/config.py` — configuration centrale

Toutes les valeurs réglables au même endroit, chacune surchargeable par
variable d'environnement (fichier `.env` chargé par `python-dotenv`).
Aucune valeur « en dur » dispersée dans le code → exigence du sujet
(« paramètres facilement modifiables »).

### `rag/loader.py` — chargement et découpage du corpus

- `load_corpus` : lit le fichier texte (UTF-8).
- `split_text` : découpe en **segments** (*chunks*). Stratégie :
  1. découpe d'abord par **paragraphes** (unités de sens naturelles) ;
  2. regroupe les paragraphes tant qu'on reste sous `CHUNK_SIZE` ;
  3. si un paragraphe dépasse `CHUNK_SIZE`, il est redécoupé en tranches
     avec un **chevauchement** (`CHUNK_OVERLAP`) pour ne pas couper une
     idée en plein milieu — le début d'une tranche répète la fin de la
     précédente.
- Pourquoi découper ? (1) Un segment = une unité de recherche : plus le
  segment est ciblé, plus la recherche est précise. (2) Le LLM a une
  fenêtre de contexte limitée : on ne peut pas lui donner tout le corpus.

### `rag/vectorstore.py` — base vectorielle

- **ChromaDB** en mémoire (`chromadb.Client()`), collection nommée
  `corpus-<taille>` (une collection par taille de segment).
- `index()` : ChromaDB calcule automatiquement l'**embedding** de chaque
  segment avec sa fonction par défaut (modèle **all-MiniLM-L6-v2** exécuté
  localement via **ONNX**, sans GPU) et stocke les vecteurs.
- `search()` : convertit la question en vecteur, calcule la **distance**
  entre ce vecteur et tous les vecteurs stockés, retourne les `top_k`
  segments les plus proches (distance la plus faible = plus similaire).
- Le bloc `pysqlite3` en tête : Streamlit Community Cloud fournit un
  sqlite3 système trop ancien pour ChromaDB ; on le remplace par le paquet
  `pysqlite3-binary` quand il est présent (Linux uniquement).

### `rag/llm.py` — appel au LLM via Ollama

- Construit le **prompt** : les passages récupérés (le « contexte »), la
  question, puis les consignes : répondre uniquement depuis les extraits,
  ne rien inventer, répondre exactement « Je ne trouve pas cette
  information dans le corpus » si l'information est absente.
- Envoie une requête HTTP `POST /api/generate` au serveur Ollama local.
- Options de génération :
  - `temperature: 0` → génération **déterministe** (le modèle choisit
    toujours le token le plus probable) : reproductible et moins de
    fantaisie, adapté à un assistant factuel ;
  - `num_predict: 512` → plafond de tokens générés : les très petits
    modèles peuvent « boucler » et générer sans fin ; ce plafond évite les
    timeouts.
- Erreurs traduites en `OllamaUnavailableError` avec un message clair
  (serveur coupé, timeout).

### `rag/pipeline.py` — orchestration

Classe `RagPipeline` : construit l'index au démarrage, puis pour chaque
question enchaîne recherche → génération → détection du statut :

- `ok` : réponse produite ;
- `hors_corpus` : le modèle a répondu que l'information est absente
  (détection par motif « je ne trouve … dans le corpus ») ;
- `ollama_indisponible` / `erreur` : échec technique.

Mesure le **temps de traitement** (`time.perf_counter`) et journalise
chaque requête (bloc `finally` : le log est écrit même en cas d'erreur).

### `rag/monitoring.py` — suivi MLOps

Une ligne **JSON** par requête dans `logs/queries.jsonl` : horodatage UTC,
question, modèle, temps, nombre de segments, statut. Format **JSONL**
(un objet JSON par ligne) : facile à relire avec pandas ou jq, s'ajoute en
fin de fichier sans réécrire l'existant. C'est la partie « observabilité »
du MLOps : on peut suivre les performances et les échecs dans le temps.

### `tests/run_tests.py` — protocole de tests du sujet

Trois questions (deux présentes dans le corpus, une absente) exécutées sur
deux configurations qui ne diffèrent que par le **nombre de segments
récupérés** (top-k 3 vs 6). Écrit un tableau Markdown
(`tests/resultats_tests.md`) : statut, temps, extrait de réponse.

### `Dockerfile` — image de la webapp

```
FROM python:3.11-slim        ← image de base légère, version figée
WORKDIR /app                 ← répertoire de travail dans le conteneur
COPY requirements.txt .      ← dépendances copiées AVANT le code :
RUN pip install ...            si le code change, cette couche reste en
                               cache → reconstructions rapides
COPY rag/ .streamlit/ app.py corpus_de_travail.txt
ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8501                  ← port Streamlit documenté
CMD ["streamlit", "run", ...]← démarrage automatique de la webapp
```

- **Ollama n'est pas dans l'image** : il tourne sur l'hôte (Codespace).
  Le conteneur le joint via `host.docker.internal` (nom DNS spécial qui
  pointe vers la machine hôte depuis un conteneur).
- Rôle de Docker ici : empaqueter l'application et ses dépendances dans
  une image **reproductible** — même environnement partout, indépendamment
  de la machine.

### `.dockerignore`

Exclut du contexte de build ce qui est inutile ou sensible : `venv/`,
`.git`, `logs/`, `.env`, PDF… Build plus rapide, image plus petite, pas de
secrets embarqués.

### `.env.example`

Modèle de configuration : liste toutes les variables avec leurs valeurs
par défaut. On le copie en `.env` (non versionné, dans `.gitignore`) pour
personnaliser sans toucher au code.

### `.streamlit/config.toml`

Thème visuel (couleur primaire, fonds) + `headless = true` (pas
d'ouverture de navigateur automatique côté serveur).

### `requirements.txt`

Les dépendances : `streamlit` (interface), `chromadb` (base vectorielle),
`requests` (HTTP vers Ollama), `python-dotenv` (lecture du `.env`),
`pysqlite3-binary` (Linux seulement, pour Streamlit Cloud).

---

## 3. Les paramètres et leurs valeurs

| Paramètre | Valeur | Pourquoi cette valeur |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Le 0.5b recommandé **hallucine** sur les questions absentes (testé : il a inventé un prix du bitcoin). Le 1.5b refuse correctement et reste léger (~1 Go), compatible Codespace. |
| `CHUNK_SIZE` | 800 caractères | Assez grand pour contenir une idée complète (un paragraphe du discours), assez petit pour rester ciblé. Trop petit → contexte fragmenté ; trop grand → bruit dans le contexte et recherche moins précise. |
| `CHUNK_OVERLAP` | 100 caractères | Évite de couper une phrase/idée à la frontière de deux segments : la fin d'un segment est répétée au début du suivant. |
| `TOP_K` | 3 | Compromis : assez de contexte pour répondre, pas trop pour ne pas noyer le petit modèle. Comparé à k=6 dans les tests (k=6 → temps un peu plus long, statuts identiques). |
| `temperature` | 0 | Déterminisme : réponses reproductibles, moins d'invention. |
| `num_predict` | 512 | Plafond anti-boucle des petits modèles. |
| `LLM_TIMEOUT` | 120 s | Marge pour une machine lente (CPU Codespace). |

---

## 4. Vocabulaire technique à maîtriser

- **RAG** (Retrieval-Augmented Generation) : architecture qui combine
  recherche documentaire + génération LLM. La réponse est « augmentée »
  par les documents récupérés.
- **LLM** (Large Language Model) : modèle de langage génératif
  (ici qwen2.5, famille Qwen d'Alibaba). Le suffixe `1.5b` = 1,5 milliard
  de **paramètres** (les poids appris du réseau de neurones).
- **Embedding** : représentation d'un texte sous forme de **vecteur** de
  nombres (ici 384 dimensions) telle que deux textes de sens proche ont
  des vecteurs proches. Produit par un modèle d'embedding
  (all-MiniLM-L6-v2), différent du LLM.
- **Base vectorielle** : base de données optimisée pour stocker des
  vecteurs et chercher les plus proches d'un vecteur requête (ChromaDB,
  FAISS, etc.).
- **Distance / similarité** : mesure de proximité entre deux vecteurs.
  ChromaDB affiche une distance (L2 au carré) : **plus elle est petite,
  plus le segment est proche de la question**. C'est la valeur affichée
  sous chaque passage dans l'interface.
- **Recherche sémantique** : recherche par le **sens** (via les vecteurs)
  et non par mots-clés exacts — « voiture » retrouve « automobile ».
- **Segment / chunk** : morceau de corpus indexé individuellement.
- **Top-k** : nombre de segments les plus proches retournés.
- **Prompt** : texte d'instruction envoyé au LLM (contexte + question +
  consignes).
- **Hallucination** : réponse inventée par le LLM, non fondée sur les
  sources. Le RAG + les consignes du prompt la limitent.
- **Température** : paramètre de génération contrôlant l'aléa. 0 =
  déterministe ; élevé = créatif/instable.
- **Token** : unité de texte du LLM (~ un mot ou morceau de mot).
  `num_predict` limite le nombre de tokens générés.
- **Ollama** : serveur local qui télécharge et exécute des LLM open source
  et les expose via une API HTTP (`localhost:11434`).
- **Inférence** : exécution du modèle (par opposition à l'entraînement).
- **ONNX** : format d'exécution portable de modèles ; permet de faire
  tourner le modèle d'embedding sur CPU sans PyTorch.
- **Image / conteneur Docker** : l'image = modèle figé (code +
  dépendances + runtime) ; le conteneur = instance en cours d'exécution
  de l'image, isolée du système hôte.
- **JSONL** : fichier texte, un objet JSON par ligne — format standard de
  journalisation.
- **Variable d'environnement / `.env`** : configuration injectée de
  l'extérieur du code (12-factor app) — permet de changer modèle, URL,
  paramètres sans modifier le code.

---

## 5. Questions probables du prof — réponses courtes

**« Expliquez le fonctionnement du RAG. »**
Deux phases. Indexation : corpus découpé en segments, chaque segment
converti en vecteur (embedding), vecteurs stockés dans ChromaDB. Question :
la question est vectorisée, on récupère les k segments les plus proches,
on les insère dans un prompt avec la question, le LLM rédige une réponse
fondée uniquement sur ces passages.

**« Quel est le rôle de Docker ? »**
Empaqueter l'application (code, dépendances, corpus, runtime Python) dans
une image reproductible : même comportement sur ma machine, le Codespace
ou un serveur. Le Dockerfile décrit la construction ; le conteneur expose
le port 8501 et démarre Streamlit automatiquement. Ollama reste hors du
conteneur, joint par le réseau.

**« Pourquoi qwen2.5:1.5b et pas 0.5b ? »**
Testé les deux. Le 0.5b hallucine sur les questions absentes du corpus
(il a inventé un prix du bitcoin) ; le 1.5b répond correctement et
signale correctement les informations absentes. ~1 Go, il reste dans le
budget mémoire du Codespace. Le nom du modèle est configurable
(`OLLAMA_MODEL`).

**« Pourquoi ce découpage (800/100) ? »**
Découpage par paragraphes regroupés jusqu'à 800 caractères : un paragraphe
du discours = une idée. Chevauchement de 100 pour ne pas couper une idée à
la frontière. Testé aussi la variation du top-k (3 vs 6) : voir tableau de
résultats.

**« Comment fonctionne la recherche vectorielle ? »**
Chaque texte devient un vecteur de 384 nombres via un modèle d'embedding ;
la proximité géométrique des vecteurs reflète la proximité de sens.
ChromaDB calcule la distance entre le vecteur de la question et ceux des
segments et retourne les k plus proches. C'est une recherche par sens,
pas par mots-clés.

**« Quelles sont les limites du projet ? »**
1. Le modèle d'embedding par défaut (MiniLM) est entraîné surtout sur de
   l'anglais ; sur notre corpus français il classe parfois mal un passage
   pertinent (constaté : la question sur les immigrés classe le bon
   passage 6e → hors du top-3 → fausse réponse « absent du corpus »).
   Piste : modèle d'embedding multilingue, ou augmenter top-k.
2. Petit LLM : reformulations parfois maladroites, sensible à la formulation.
3. Index en mémoire : réindexé à chaque démarrage (acceptable : petit corpus).
4. Détection « hors corpus » par motif textuel : robuste mais pas parfaite.
5. Suivi MLOps minimal (fichier local) : pas de dashboard, pas d'alerte.

**« Que se passe-t-il si Ollama est coupé ? »**
L'appel HTTP échoue, l'exception est traduite en `OllamaUnavailableError`,
l'interface affiche un message clair, et la requête est journalisée avec
le statut `ollama_indisponible`.

**« Pourquoi la température à 0 ? »**
Assistant factuel : on veut la réponse la plus probable et reproductible,
pas de créativité. Aussi indispensable pour comparer équitablement deux
configurations dans les tests.

**« Comment l'application sait-elle que l'information est absente ? »**
Le prompt impose une phrase de refus exacte ; le pipeline détecte le motif
« je ne trouve … dans le corpus » dans la réponse et marque le statut
`hors_corpus`, affiché comme bandeau d'information.

---

## 6. Chiffres à retenir

- Corpus : 27 segments (taille 800, chevauchement 100).
- Embeddings : all-MiniLM-L6-v2, 384 dimensions, local (ONNX/CPU).
- LLM : qwen2.5:1.5b (~1 Go), servi par Ollama, température 0.
- Temps constatés en local : ~0,5–3 s par question ; premier appel plus
  lent (chargement du modèle).
- Tests : 6/6 statuts corrects (2 configs × 3 questions).
