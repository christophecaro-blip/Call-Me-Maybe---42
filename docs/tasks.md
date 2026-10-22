# Tasks — call me maybe

> Feuille de route pas à pas. Chaque phase produit quelque chose qui **fonctionne** et que tu
> **comprends** avant de passer à la suivante.
> Format : 🎯 objectif · 🔧 étapes · 🧠 à comprendre (renvoi vers [concepts.md](concepts.md)) · ✅ terminé quand.
> Structure cible : voir [architecture.md](architecture.md).

Vue d'ensemble

| Phase | Thème | Résultat |
|---|---|---|
| 0 | Mise en place | `uv sync`, `make lint` passent sur un projet vide |
| 1 | Exploration du SDK et du tokenizer | tu connais les vraies données que tu manipules |
| 2 | Modèles de données & I/O | lecture/écriture robustes, sans LLM |
| 3 | Vocabulaire | `id → texte` fiable |
| 4 | Moteur de décodage + choix de fonction | 1er résultat réel : le bon nom de fonction |
| 5 | Contraintes de valeurs | nombres, booléens, strings |
| 6 | Pipeline complet | fichier de sortie complet et valide |
| 7 | Qualité des réponses | ≥ 90 % de précision mesurée |
| 8 | Performance | < 5 min, mesuré |
| 9 | Robustesse | aucun crash sur entrées hostiles |
| 10 | Finition | lint strict, docstrings, README |
| 11 | Préparation soutenance | tu sais tout expliquer et modifier en direct |

---

## Phase 0 — Mise en place du projet

🎯 Un squelette qui s'installe, se lance et passe le lint.

🔧 Étapes
- [ ] Copier `attachments_project_ai/llm_sdk/` → `./llm_sdk/` et
      `attachments_project_ai/data/input/` → `./data/input/`. Ensuite, supprimer
      `attachments_project_ai/` (ou l'ajouter au `.gitignore`) pour ne pas avoir de doublon dans le dépôt.
- [ ] `uv init` avec `requires-python = ">=3.10"` ; `uv add numpy pydantic` ;
      dev : `uv add --dev flake8 mypy pytest`.
- [ ] Déclarer le SDK : `uv add --editable ./llm_sdk` (ce qui écrit le bloc `[tool.uv.sources]`
      de concepts §11). Vérifier le `pyproject.toml` obtenu.
- [ ] Lire `llm_sdk/llm_sdk/__init__.py` en entier et lister ses méthodes **publiques**
      (tableau dans architecture §5).
- [ ] Créer l'arborescence `src/` de `architecture.md` avec des modules vides + `__main__.py`
      qui affiche juste « ok ».
- [ ] `.gitignore` : `.venv/`, `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `data/output/`.
- [ ] `.flake8` + `[tool.mypy]` : exclure `.venv` et `llm_sdk`.
- [ ] `Makefile` : `install`, `run`, `debug`, `clean`, `lint`, `lint-strict` (flags exacts du sujet).
- [ ] Premier commit.

🧠 concepts §11.

✅ Terminé quand : dans un **clone vierge**, `uv sync && make run && make lint` réussissent.

---

## Phase 1 — Explorer le SDK et le tokenizer (scripts jetables, hors `src/`)

🎯 Voir de tes yeux ce que sont tokens, ids, logits. Cette phase évite 80 % des mauvaises surprises.

🔧 Étapes (dans un dossier `scratch/` non commité, ou un notebook)
- [ ] Ajouter `scratch/` au `.gitignore` et à l'`extend-exclude` de `.flake8` (sinon `make lint` l'analyse).
- [ ] Un petit script par expérience (`01_load.py`, `02_generate.py`, `03_vocab.py`…) plutôt qu'un
      seul `test.py` où l'on commente et décommente des blocs : chacun reste lisible et relançable.
- [ ] Noter les réponses au fur et à mesure dans `scratch/notes.md` (ce sont les réponses au
      critère « Terminé quand… » en fin de phase, et de la matière pour le README).
- [ ] Instancier `Small_LLM_Model()` ; mesurer le **temps de chargement** (détail ci-dessous).
- [ ] `encode("What is the sum of 265 and 345?")` renvoie un tensor `[1, n]` → vérifier que
      `encode(t)[0].tolist()` donne bien une `list[int]`. (détail ci-dessous)
- [ ] Vérifier que `encode("<|im_start|>user")` produit **un seul** id pour `<|im_start|>`
      (sinon le chat template serait mal tokenisé).
- [ ] Ouvrir `vocab.json` (`get_path_to_vocab_file()`) et `tokenizer.json`
      (`get_path_to_tokenizer_file()`, clé `added_tokens`) : nombre d'entrées de chacun ?
      (détail ci-dessous)
- [ ] Afficher token par token : retrouver `Ġ`, vérifier que les nombres sont découpés chiffre par chiffre.
- [ ] Chercher des tokens « mixtes » : lister tous les tokens contenant `"` → combien ? lesquels ?
      (détail ci-dessous)
- [ ] `get_logits_from_input_ids(ids)` : longueur du vecteur ? **= taille du vocab ?** (noter l'écart ;
      détail ci-dessous).
- [ ] Écrire une boucle **greedy non contrainte** de 40 tokens avec un prompt « réponds en JSON » :
      observer le résultat (thinking ? prose ? JSON cassé ?). Garde la sortie : elle servira
      dans le README (« Challenges » : pourquoi le prompting seul ne suffit pas).
- [ ] Refaire avec le chat template Qwen3 + `<think>\n\n</think>\n\n` : différence ?
- [ ] Mesurer le temps d'un appel de logits pour un prompt de 50 puis 400 tokens.

### Étape 1 en détail : instancier `Small_LLM_Model()`

**« Instancier », c'est créer un objet à partir de la classe.** `Small_LLM_Model` est une
**classe** : un plan qui décrit ce qu'un modèle sait faire (`encode`,
`get_logits_from_input_ids`…). L'instancier, c'est fabriquer un objet concret à partir de ce plan :

```python
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()   # ← instanciation
```

Les parenthèses déclenchent la méthode `__init__` de la classe. Ensuite, `model` est un objet
« vivant » : le modèle est chargé en mémoire et prêt à répondre.

**Ce qui se passe pendant ces quelques secondes** (voir `__init__` dans
[llm_sdk/llm_sdk/__init__.py](../llm_sdk/llm_sdk/__init__.py)) :

| Étape | Ce que fait le code | Concrètement sur cette machine |
|---|---|---|
| 1. Choix du matériel | essaie `mps` (Mac), puis `cuda`, puis `cpu` | **`cuda`** : RTX 4050 |
| 2. Choix de la précision | `float16` sur GPU, `float32` sur CPU | **`float16`** : 2 octets par nombre |
| 3. Tokenizer | `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")` | télécharge `vocab.json`, `merges.txt`, `tokenizer.json`… depuis Hugging Face |
| 4. Poids du modèle | `AutoModelForCausalLM.from_pretrained(...)` | télécharge environ 1,5 Go de poids (fichier `.safetensors`) |
| 5. Mise en mémoire | `self._model.to("cuda")` | ~0,6 milliard de paramètres × 2 octets ≈ **1,2 Go de mémoire GPU** |
| 6. Mode inférence | `eval()` et `requires_grad = False` | on ne va pas entraîner le modèle, seulement l'utiliser : PyTorch ne garde pas ce qui sert à l'entraînement |

- **Le premier lancement télécharge tout**, ce qui peut prendre plusieurs minutes. Les fichiers vont
  dans `$XDG_CACHE_HOME/huggingface` : à 42, `XDG_CACHE_HOME=/goinfre/ccaro/.cache`, donc
  `/goinfre/ccaro/.cache/huggingface` (sans cette variable, ce serait `~/.cache/huggingface`).
- **Les lancements suivants lisent le cache local** et prennent quelques secondes.
- C'est pour ça que le programme final instanciera le modèle **une seule fois** pour tous les prompts.

**En pratique**, crée `scratch/01_load.py` :

```python
import time

from llm_sdk import Small_LLM_Model

start = time.perf_counter()
model = Small_LLM_Model()
elapsed = time.perf_counter() - start

print(f"Modèle chargé en {elapsed:.1f} s")
print(type(model))
```

Lance-le depuis la racine du projet :

```bash
uv run python scratch/01_load.py
```

- [ ] Le lancer **deux fois** et comparer : le 1er temps inclut le téléchargement, le 2e est le vrai
      temps de chargement (à noter pour « Performance analysis » du README).
- [ ] Pendant que le script tourne (ajouter un `input()` à la fin), lancer `nvidia-smi` dans un autre
      terminal : environ 1,2 Go de mémoire GPU occupés par `python`.
- [ ] Après le 1er lancement, `ls $XDG_CACHE_HOME/huggingface/hub/` : un dossier `models--Qwen--Qwen3-0.6B`
      contient les fichiers téléchargés.

**À savoir dire en soutenance :** « À l'instanciation, le SDK choisit le GPU s'il y en a un, charge
le tokenizer et les poids de Qwen3-0.6B depuis le cache Hugging Face (ou les télécharge la première
fois), les place en mémoire GPU en float16 et passe le modèle en mode inférence. C'est l'opération
la plus coûteuse du programme, donc je ne la fais qu'une fois. »

### De `encode` aux logits : ce qui se passe vraiment

Ces deux méthodes sont les deux extrémités du modèle : `encode` transforme le texte en nombres
que le modèle comprend, `get_logits_from_input_ids` renvoie ce que le modèle « pense » de la suite.

```
"What is the sum of 4 and 5 ?"
        │  encode  (tokenizer, pas de réseau de neurones : rapide)
        ▼
tensor([[3838, 374, 279, 2629, 315, 220, 19, ...]])      ← forme [1, n]
        │  [0].tolist()
        ▼
[3838, 374, 279, 2629, 315, 220, 19, ...]               ← list[int], n ids
        │  get_logits_from_input_ids  (réseau de neurones : le calcul coûteux)
        ▼
[-3.1, 0.4, 12.7, ..., -8.2]                            ← UN vecteur, ~152 000 scores
        │  argmax / top-k
        ▼
id du token suivant le plus probable  →  decode  →  " The"
```

(ids illustratifs : affiche les tiens.)

#### `encode(text)` : du texte aux ids

1. Le tokenizer découpe le texte en **tokens** (morceaux de mots, ponctuation, chiffres un par un).
2. Chaque token est remplacé par son **id**, sa position dans `vocab.json`.
3. Le SDK renvoie un **tensor torch de forme `[1, n]`** : 1 ligne (le *batch*, c'est-à-dire le
   nombre de phrases traitées en même temps ; ici toujours une), `n` colonnes (les ids).
   - `[0]` sélectionne l'unique phrase du batch → tensor de forme `[n]` ;
   - `.tolist()` le convertit en `list[int]`, le format attendu par `get_logits_from_input_ids`
     et le seul que ton code manipulera (torch est interdit dans `src/`).

`encode` n'utilise pas le réseau de neurones : c'est une simple consultation de table, quasi
instantanée.

#### `get_logits_from_input_ids(ids)` : des ids à la prédiction

- **Entrée :** toute la séquence d'ids.
- **Sortie :** **un seul** vecteur de ~152 000 nombres, un **logit** par token du vocabulaire.
- **Sens :** le logit `i` mesure à quel point le token `i` est un bon **token suivant** pour cette
  séquence. Plus il est grand, plus le modèle « veut » écrire ce token. Ce ne sont pas des
  probabilités (ils peuvent être négatifs, ne somment pas à 1) ; le softmax les transformerait en
  probabilités, mais pour choisir le meilleur, l'argmax des logits suffit.

> ⚠️ **Erreur classique :** croire qu'on obtient un vecteur par token d'entrée.
> ```python
> for _ in ids:                                      # ✗ n appels identiques
>     logits.append(model.get_logits_from_input_ids(ids))
> ```
> Même entrée → même sortie, n fois. Un appel = une prédiction du token **suivant**.

#### Générer = répéter « prédire, choisir, ajouter »

Le modèle ne prédit qu'un token à la fois. Pour générer du texte, on boucle :

```python
for _ in range(max_tokens):
    logits = model.get_logits_from_input_ids(ids)   # 1. prédire
    next_id = int(np.argmax(logits))                # 2. choisir (greedy) : id du meilleur score
    ids.append(next_id)                             # 3. ajouter, puis recommencer
```

(Détail de `np.argmax`, `np.argsort` et des slices dans concepts.md §11, « numpy ».)

C'est **à l'étape 2** qu'interviendra le décodage contraint : avant l'argmax, on mettra à `-inf`
les logits des tokens qui casseraient le JSON. Tout le projet se joue à cet endroit.

Conséquence sur la performance : le SDK n'a pas de cache, donc chaque appel **relit toute la
séquence**, prompt compris. Un pas avec un prompt de 400 tokens coûte plus cher qu'avec 50
(étape « mesurer » ci-dessus).

#### Script d'exploration : `scratch/test.py`

```python
import time

import numpy as np
from llm_sdk import Small_LLM_Model


def main() -> None:
    start = time.perf_counter()
    model = Small_LLM_Model()
    print(f"Chargement : {time.perf_counter() - start:.1f} s")

    ids = model.encode("What is the sum of 4 and 5 ?")[0].tolist()
    print(f"ids ({len(ids)} tokens) : {ids}")

    start = time.perf_counter()
    logits = np.array(model.get_logits_from_input_ids(ids))  # list[float] → tableau numpy
    print(f"1 pas : {time.perf_counter() - start:.3f} s")
    print(f"Nombre de logits : {len(logits)}")

    # Les 5 tokens que le modèle juge les plus probables pour la suite
    # argsort : ids triés par score croissant ; [::-1] : décroissant ; [:5] : les 5 premiers
    top5 = np.argsort(logits)[::-1][:5]
    for token_id in top5:
        print(token_id, repr(model.decode([int(token_id)])), logits[token_id])


if __name__ == "__main__":
    main()
```

- [ ] Lancer le script : noter le nombre d'ids, la durée d'un pas, le nombre de logits.
- [ ] Lire les 5 meilleurs candidats : sont-ils cohérents comme suite de la phrase ?
- [ ] Ajouter `top5[0]` à `ids`, rappeler `get_logits_from_input_ids`, répéter 10 fois : c'est ta
      première boucle de génération. Que répond le modèle spontanément ?
- [ ] Afficher **seulement la réponse générée**, en texte : mémoriser `prompt_len = len(ids)` avant
      la boucle, puis `model.decode(ids[prompt_len:])`.
      ⚠️ `[model.decode(i) for i in ids]` décode chaque id séparément et donne une **liste** de
      morceaux ; `model.decode(ids)` décode toute la séquence et donne directement le **texte**.

**À savoir dire en soutenance :** « `encode` transforme le texte en une liste d'ids via le
tokenizer. `get_logits_from_input_ids` fait passer toute la séquence dans le réseau et renvoie un
score par token du vocabulaire pour le token suivant. Générer, c'est répéter : prédire, choisir,
ajouter. Mon décodage contraint agit entre la prédiction et le choix. »

### Explorer `vocab.json` et `tokenizer.json`

**Ne pas faire `print(f.read())`** : `tokenizer.json` fait environ 11 Mo, c'est illisible dans un
terminal. Un JSON s'explore en le **chargeant** avec `json.load`, puis en regardant sa structure
petit à petit.

`scratch/03_vocab.py` :

```python
import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

with open(model.get_path_to_tokenizer_file(), encoding="utf-8") as f:
    tok = json.load(f)

print(tok.keys())                 # 1. quelles sont les grandes sections ?
print(len(tok["added_tokens"]))   # 2. combien de tokens spéciaux ?
print(tok["added_tokens"][:3])    # 3. à quoi ressemble une entrée ?

with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
    vocab: dict[str, int] = json.load(f)

print(len(vocab), max(vocab.values()))
```

Questions à résoudre :
- [ ] Combien d'entrées dans `vocab.json` ? Quel est le plus grand id ?
- [ ] Combien de `added_tokens` ? Quels sont leurs ids, et sont-ils dans `vocab.json` ?
- [ ] Additionner `len(vocab)` (question 1) et `len(tok["added_tokens"])` (question 2) : c'est le
      nombre total de tokens connus du tokenizer. Le comparer avec `len(logits)` (taille du vecteur
      renvoyé par `get_logits_from_input_ids`, quel que soit le prompt). Que représentent les ids
      en trop, qui ont un score mais aucun token ? (La somme n'a de sens que si la question 2 a
      montré que les tokens spéciaux ne sont pas déjà dans `vocab.json`.)
- [ ] Retrouver dans `added_tokens` les ids de `<|im_start|>`, `<|im_end|>`, `<think>` et
      `</think>` : ils serviront pour le format de conversation (étape suivante).

**Tokens mixtes**, dans le même script :

```python
mixed = [t for t in vocab if '"' in t]
print(len(mixed), mixed[:30])
```

- [ ] Combien de tokens contiennent `"` ?
- [ ] Lesquels mélangent `"` avec `,`, `}` ou `:` ? Noter 2 ou 3 exemples : ce sont eux que le
      décodeur contraint devra gérer (concepts §5 et §8).
- [ ] Compter, pour chacun de ces caractères, combien de tokens le contiennent **en plus** de `"` :
      ```python
      for c in [",", "}", ":"]:
          found = sorted([t for t in mixed if c in t], key=len)
          print(c, len(found), found[:12])
      ```
      Repérer aussi `Ġ"` (espace + guillemet) et `\"` (guillemet échappé).
      **Leçon :** le vocabulaire contient des centaines de tokens qui collent un guillemet à de la
      ponctuation JSON ; le modèle les utilise volontiers, car un token fait le travail de plusieurs
      caractères.

**Exercice : le JSON reste-t-il valide ?**

Situation : le décodeur génère la valeur d'une string et a déjà écrit `"shrek`. Le modèle propose
un token mixte. Est-ce que le JSON reste valide ?

Exemple résolu :

| Token proposé | Texte obtenu | Valide ? | Pourquoi |
|---|---|---|---|
| `'":'` | `"shrek":` | ❌ | `"` ferme la string, mais `:` est interdit après une **valeur** en JSON : on n'en trouve qu'après une **clé** |

- [ ] Remplir ce tableau dans `scratch/notes.md` :

| Token proposé | Texte obtenu | Valide ? | Pourquoi |
|---|---|---|---|
| `'"'` | | | |
| `':"'` | | | |
| `'",'` | | | |
| `'"}'` | | | |
| `'\\"'` (`\"`) | | | |
| `'Ġ"'` | | | |

Indices :
- **Ce qui suit la valeur dépend de la position.** Si `name` est le dernier paramètre de
  `fn_greet`, le JSON attendu après la valeur est `"}}`. S'il y a un autre paramètre après, c'est
  `", "...`.
- **`\"` ne ferme pas la string** : c'est un guillemet *dans* le texte.
- Certains tokens sont **valides seulement selon ce qui doit venir ensuite** : ce sont les plus
  intéressants (concepts §8, ligne « Token `5,` / `"}` / `}}` »).

✅ Exercice réussi quand tu peux énoncer la règle : un token est valide si et seulement si **tous**
ses caractères respectent la grammaire, un par un (concepts §5). C'est la règle centrale du futur
décodeur.

**À savoir dire en soutenance :** « `vocab.json` associe chaque token BPE à son id ; les tokens
spéciaux comme `<|im_start|>` sont à part, dans les `added_tokens` de `tokenizer.json`. Le modèle
produit plus de logits qu'il n'y a de tokens réels : les ids sans token doivent être interdits.
Beaucoup de tokens mélangent un guillemet et de la ponctuation JSON : un token n'est accepté que si
chacun de ses caractères respecte la grammaire à la position courante. »

🧠 concepts §2, §3, §4, §9.

✅ Terminé quand tu peux répondre de mémoire : taille du vocab, taille des logits, durée d'un pas,
forme de `encode`, signification de `Ġ`, exemple de token mixte.

---

## Phase 2 — Modèles de données, CLI et I/O (sans LLM)

🎯 Tout le « plombier » du programme, testable sans charger le modèle.

🔧 Étapes
- [ ] `models.py` : `ParameterSpec`, `FunctionDefinition`, `PromptItem`, `FunctionCall`
      (pydantic). Calquer sur le **vrai** `functions_definition.json`.
- [ ] `errors.py` : `CallMeMaybeError` (base) → `InputFileError`, `ModelError`, `DecodingError`.
- [ ] `cli.py` : argparse avec `--functions_definition`, `--input`, `--output` et leurs valeurs
      par défaut (`data/input/functions_definition.json`, `data/input/function_calling_tests.json`,
      `data/output/function_calling_results.json`) → `CliConfig` pydantic.
- [ ] `io_utils.py` :
  - `load_functions(path) -> list[FunctionDefinition]`
  - `load_prompts(path) -> list[PromptItem]`
  - `write_results(path, calls) -> None` (crée le dossier parent, `json.dump(..., indent=2, ensure_ascii=False)`)
  - chaque erreur (`FileNotFoundError`, `JSONDecodeError`, `ValidationError`, `OSError`)
    convertie en `InputFileError` avec un message lisible.
- [ ] `__main__.py` : `main() -> int` ; un `try/except CallMeMaybeError` global → message sur
      stderr, code 1 ; `KeyboardInterrupt` → code 130.
- [ ] Brancher un **faux** pipeline : pour chaque prompt, sortir la 1re fonction avec des valeurs
      bidon → vérifie toute la chaîne entrée → sortie.
- [ ] Tests pytest : fichier absent, JSON invalide, schéma invalide, liste vide.

🧠 concepts §10.

✅ Terminé quand `make run` produit un fichier de sortie (faux contenu, vrai format) et que
chaque cas d'erreur affiche un message clair sans traceback.

---

## Phase 3 — Vocabulaire

🎯 Pour tout id, connaître le **texte réel** qu'il ajoute à la sortie.

🔧 Étapes
- [ ] `llm/vocabulary.py` : classe `Vocabulary` qui charge `vocab.json` (+ les `added_tokens`
      de `tokenizer.json` pour connaître les ids spéciaux) et construit
      `id_to_text: list[str | None]` (taille = taille des logits ; `None` = id inutilisable).
- [ ] Implémenter le décodage byte-level (concepts §3) **ou** le cache via `decode([id])`.
      Comparer les deux sur 1000 ids au hasard : mêmes résultats ?
- [ ] Marquer les tokens spéciaux et les fragments UTF-8 invalides comme inutilisables.
- [ ] `llm/engine.py` : classe `Engine` (façade sur le SDK) avec
      `encode(text) -> list[int]` et `next_logits(ids) -> np.ndarray`. Tout le reste du code
      ne parle qu'à `Engine`, jamais directement au SDK.
- [ ] Tests : `"Ġthe"` → `" the"`, `"Ċ"` → `"\n"`, un chiffre, un token spécial.

🧠 concepts §3.

✅ Terminé quand `"".join(id_to_text[i] for i in encode(s)) == s` pour une dizaine de phrases
(ASCII, chiffres, ponctuation, guillemets, retours ligne).

---

## Phase 4 — Moteur de décodage contraint + choix de la fonction

🎯 Premier vrai résultat : **le LLM choisit le nom de fonction**, avec garantie de validité.

🔧 Étapes
- [ ] `decoding/constraints.py` : interface `Constraint` (`accepts(text)`, `advance(text)`,
      `is_complete()`) — cf. architecture D2.
- [ ] `ChoiceConstraint(options: list[str])` : accepte un texte si `généré + texte` est préfixe
      d'une option ; se termine quand une option est complète **et** que le `"` fermant arrive.
      Tests unitaires (sans modèle !) : préfixes, option complète, options qui partagent un préfixe
      (`fn_add` / `fn_add_numbers` : comment tu tranches ?).
- [ ] `decoding/decoder.py` : `decode_value(ids, constraint, max_tokens) -> (ids, texte)` :
  1. logits = engine.next_logits(ids)
  2. parcourir les ids par logit décroissant ; garder le premier dont le texte est accepté
  3. `advance`, ajouter l'id, recommencer jusqu'à `is_complete()` ou `max_tokens`.
- [ ] `prompting.py` : `build_prompt(functions, user_prompt) -> str` (chat template, thinking désactivé).
- [ ] Brancher : forcer `{"name": "` puis `decode_value(ChoiceConstraint(noms))`.
- [ ] Afficher pour chaque prompt du jeu de test : nom choisi + les 5 meilleurs candidats valides
      avec leurs logits au 1er pas (super utile pour comprendre et pour le README).

🧠 concepts §5, §6.

✅ Terminé quand tous les prompts de test reçoivent un nom de fonction **existant** (garanti),
et que tu mesures combien sont **corrects**.

---

## Phase 5 — Contraintes de valeurs

🎯 Une contrainte par type, chacune testée **sans le modèle**.

🔧 Étapes (une à la fois, tests d'abord)
- [ ] `BooleanConstraint` → réutiliser `ChoiceConstraint(["true", "false"])` + marqueur de fin.
- [ ] `NumberConstraint(end_chars)` : automate de concepts §7, longueur max, fin autorisée
      seulement en état acceptant. Tests : `0`, `-3`, `3.14`, `007` (refusé), `1.` (incomplet),
      `-` (incomplet), token `"5,"`.
- [ ] `IntegerConstraint` : idem sans `.` (si le type existe dans les définitions).
- [ ] `StringConstraint` v1 : tout caractère sauf `"`, `\` et contrôles ; fin sur `"`.
      Tests : vide, espaces, apostrophes, unicode.
- [ ] `StringConstraint` v2 : échappements (`\\`, `\"`, `\n`, `\uXXXX`). Test : `\d+` doit
      sortir en JSON `"\\d+"` et redonner `\d+` après `json.loads`.
      **Obligatoire** : le prompt « Replace all numbers in … » du jeu fourni en a besoin.
- [ ] Décider la politique des tokens qui débordent après le marqueur de fin (concepts §8)
      et l'écrire dans le README (« Design decisions »).
- [ ] Fabrique : `make_constraint(spec: ParameterSpec) -> Constraint` ; type inconnu → erreur claire.

🧠 concepts §7, §8.

✅ Terminé quand `pytest tests/test_constraints.py` passe et couvre chaque état de chaque automate.

---

## Phase 6 — Pipeline complet

🎯 Le vrai fichier de sortie, valide à 100 %.

🔧 Étapes
- [ ] `pipeline.py` : `FunctionCaller` (pydantic, contient `Engine`, `Vocabulary`, définitions).
      `call(prompt) -> FunctionCall` :
  1. prompt → ids
  2. forcer `{"name": "` → choix → forcer `", "parameters": {`
  3. pour chaque paramètre : forcer `"<nom>": ` (+ `"` si string) → `decode_value` → forcer `, ` ou rien
  4. forcer `}}`
  5. `json.loads(texte_généré)` → conversion des types (`number` → `float`) → `FunctionCall`
- [ ] Optimisation immédiate : les segments forcés sont ajoutés **sans** appel au modèle.
- [ ] Fonction sans paramètre : `"parameters": {}` — tester.
- [ ] `try/except` par prompt : un échec → log stderr + on continue (décider : on omet l'entrée ?
      on écrit quand même un appel ? → justifier).
- [ ] Remplacer le faux pipeline de la phase 2.

🧠 concepts §6, architecture D1/D5.

✅ Terminé quand `make run` produit `data/output/function_calling_results.json`, que
`python -c "import json; json.load(open(...))"` passe, et que chaque entrée a exactement
les clés `prompt`, `name`, `parameters` avec les bons types.

---

## Phase 7 — Qualité des réponses (≥ 90 %)

🎯 Mesurer, puis améliorer la justesse — pas seulement la validité.

🔧 Étapes
- [ ] Écrire à la main un fichier `tests/expected.json` (prompt → réponse attendue). Pour les
      3 prompts regex, plusieurs réponses peuvent être justes (`\d+` ou `[0-9]+`,
      `[aeiouAEIOU]` ou `[aeiou]`) : ton script doit le tolérer, par exemple en **appliquant**
      la regex obtenue avec `re.sub` et en comparant le résultat au résultat attendu.
      L'appel à `fn_greet` doit donner `{"name": "shrek"}` et la somme `{"a": 2.0, "b": 3.0}`.
- [ ] Script d'évaluation : % nom correct, % arguments corrects, liste des erreurs.
- [ ] **Élargir le jeu de test** avec tes propres prompts et fonctions (le correcteur changera les
      fichiers !) : multi-paramètres, booléens, strings avec guillemets, nombres négatifs/décimaux,
      grands nombres, prompts ambigus, formulations variées.
- [ ] Itérer sur le prompt système (une variable à la fois, mesurer à chaque fois) :
      format de la liste des fonctions, descriptions des paramètres, 1–2 exemples génériques.
- [ ] Optionnel : choix de la fonction par **score de séquence** (somme des log-softmax) vs
      greedy sur trie — comparer.
- [ ] Noter chaque essai et son score : matière directe pour « Performance analysis » et « Challenges ».

🧠 concepts §4, §5.

✅ Terminé quand ≥ 90 % sur le jeu fourni **et** sur ton jeu élargi.

---

## Phase 8 — Performance (< 5 min)

🎯 Rapide, et prouvé par des mesures.

🔧 Étapes
- [ ] Instrumenter : temps de chargement, temps par prompt, nb d'appels modèle, nb de tokens forcés.
- [ ] Recherche paresseuse des candidats (`argsort`) si ce n'est pas déjà le cas.
- [ ] Précalculer les masques/ensembles utiles (tokens chiffres, tokens « sûrs » pour une string…).
- [ ] Raccourcir le prompt système sans perdre en précision (re-mesurer phase 7 !).
- [ ] Option `--verbose` / logs de timing désactivés par défaut.

🧠 concepts §9.

✅ Terminé quand le jeu complet passe en < 5 min sur ta machine, chiffres notés.

---

## Phase 9 — Robustesse

🎯 Le correcteur va essayer de casser ton programme. Il ne doit pas y arriver.

🔧 Checklist de tests manuels (à scripter si possible)
- [ ] fichier de fonctions / de prompts absent
- [ ] JSON invalide (virgule en trop, fichier vide, fichier binaire)
- [ ] JSON valide mais mauvaise forme (objet au lieu de liste, clé `prompt` manquante, type inconnu)
- [ ] liste de prompts vide → sortie `[]` ? message ? (décider et justifier)
- [ ] prompt vide `""`, prompt très long, caractères spéciaux / unicode / guillemets
- [ ] prompt sans rapport avec aucune fonction (« what's the weather? ») → une fonction est quand même
      choisie (le format l'impose) : sache l'expliquer
- [ ] dossier de sortie inexistant, chemin de sortie non inscriptible
- [ ] fonction avec 0, 1, 4+ paramètres ; noms de fonctions qui partagent un préfixe
- [ ] `Ctrl+C` pendant le traitement

✅ Terminé quand aucun cas ne produit de traceback et que chaque message est compréhensible.

---

## Phase 10 — Finition

🔧 Étapes
- [ ] `make lint` sans erreur ; viser `make lint-strict`.
- [ ] Docstrings PEP 257 (style Google) sur **chaque** fonction et classe.
- [ ] Vérifier : aucune classe non pydantic (hors exceptions — justifier), aucun membre privé du SDK
      utilisé, aucun import interdit (torch/transformers/outlines… dans `src/`).
- [ ] `README.md` en anglais, **1re ligne en italique** :
      `*This activity has been created as part of the 42 curriculum by <login>.*`
      Sections : Description, Instructions, Resources (+ usage de l'IA), Algorithm explanation,
      Design decisions, Performance analysis, Challenges faced, Testing strategy, Example usage.
- [ ] `data/output/` absent du dépôt ; `uv.lock` présent.
- [ ] Test final dans un clone vierge : `uv sync && uv run python -m src`.

✅ Terminé quand le dépôt est conforme à la checklist du chapitre 7 du sujet.

---

## Phase 11 — Préparation de la soutenance

🎯 Prouver que tu comprends, et être capable de modifier en quelques minutes.

🔧 Étapes
- [ ] Relire toutes les sections « ✅ Tu dois savoir expliquer » de `concepts.md` et les expliquer
      à voix haute (ou à un pair — c'est ce que recommande le sujet).
- [ ] Savoir faire en direct un pas de génération « à la main » : prompt → logits → masque → token.
- [ ] S'entraîner aux modifications probables :
  - [ ] ajouter un nouveau type de paramètre (ex. `integer` si absent, ou une liste de nombres) ;
  - [ ] renommer une clé de sortie / ajouter un champ ;
  - [ ] passer de greedy à un sampling avec température ;
  - [ ] afficher les k meilleurs candidats à chaque pas ;
  - [ ] changer la longueur max des strings ;
  - [ ] utiliser un autre modèle Qwen.
- [ ] Préparer une démo : run normal, run avec fichier cassé, run avec tes propres fonctions.

✅ Terminé quand chaque modification ci-dessus te prend moins de 5 minutes.
