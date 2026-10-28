# Notes — Phase 1 : exploration du SDK et du tokenizer

> Suit l'ordre des étapes de `docs/tasks.md`, phase 1.
> Script associé indiqué à chaque section. Les `⏳ À compléter` sont les trous restants.

## Sommaire

1. [Chargement du modèle](#1-chargement-du-modèle) — `01_load.py`
2. [vocab.json, tokenizer.json et logits](#2-vocabjson-tokenizerjson-et-logits) — `03_vocab.py`
3. [Tokens mixtes](#3-tokens-mixtes) — `03_vocab.py`
4. [Exercice : le JSON reste-t-il valide ?](#4-exercice--le-json-reste-t-il-valide-) — `04_json_check.py`
5. [Token spécial `<|im_start|>`](#5-token-spécial-im_start) — `05_special_tokens.py`
6. [Découpage : `Ġ` et chiffres](#6-découpage--ġ-et-chiffres) — `06_tokens.py`
7. [Génération non contrainte](#7-génération-non-contrainte) — `02_generate.py`
8. [Temps d'un pas](#8-temps-dun-pas) — `02_generate.py`
9. [Récapitulatif « Terminé quand… »](#9-récapitulatif--terminé-quand-)

---

## 1. Chargement du modèle

- **Temps de chargement : ~2,1 s** (modèle déjà dans le cache Hugging Face, CPU).
  Coût payé **une seule fois** par exécution : le programme final n'instancie le modèle qu'une fois.
- **Matériel réellement utilisé : CPU.** `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`.
  Sous Windows, le `torch` de PyPI est la version CPU : le SDK se rabat donc sur le CPU en
  `float32`, malgré la RTX 4050. Avantage : les mesures sont représentatives d'un poste sans GPU.
  → À citer dans « Performance analysis » du README.

---

## 2. vocab.json, tokenizer.json et logits

| Question | Réponse |
|---|---|
| Entrées dans `vocab.json` / plus grand id | **151 643** / **151 642** |
| `added_tokens` (dans `tokenizer.json`) | **26**, ids **151 643 → 151 668**, absents de `vocab.json` |
| Total tokens connus | 151 643 + 26 = **151 669** |
| `len(logits)` | **151 936** |

**Écart logits / tokens :** 151 936 − 151 669 = 267 logits sans token associé. La couche de
sortie est dimensionnée en multiple de 128 (151 936 = 1 187 × 128), plus efficace pour le calcul.
Ces logits n'ont pas de sens et doivent être **interdits**.

**Ids utiles :**

| Token | Id |
|---|---|
| `<|im_start|>` | 151644 |
| `<|im_end|>` | 151645 |
| `<think>` | 151667 |
| `</think>` | 151668 |

### Champ `special` des 26 `added_tokens` (`03_vocab.py`)

Aucun des 26 n'est dans `vocab.json` (ni par id, ni par texte).

| Ids | Tokens | `special` |
|---|---|---|
| 151643 – 151645 | `<|endoftext|>` · `<|im_start|>` · `<|im_end|>` | **True** |
| 151646 – 151656 | `<|object_ref_start|>` · `<|object_ref_end|>` · `<|box_start|>` · `<|box_end|>` · `<|quad_start|>` · `<|quad_end|>` · `<|vision_start|>` · `<|vision_end|>` · `<|vision_pad|>` · `<|image_pad|>` · `<|video_pad|>` | **True** |
| 151657 – 151668 | `<tool_call>` · `</tool_call>` · `<|fim_prefix|>` · `<|fim_middle|>` · `<|fim_suffix|>` · `<|fim_pad|>` · `<|repo_name|>` · `<|file_sep|>` · `<tool_response>` · `</tool_response>` · **`<think>`** · **`</think>`** | **False** |

→ **14 True** (151643 – 151656), **12 False** (151657 – 151668).

**Conséquences :**
- `special: True` → `decode` (`skip_special_tokens=True`) les **masque** (`''`). D'où `''` pour
  `<|im_start|>` en section 5.
- `special: False` → `decode` les **affiche** comme du texte. D'où `<think>` visible dans
  l'expérience 2 (section 7).
- **Règle pour la phase 3 :** interdire **tous** les `added_tokens` pendant la génération, quelle
  que soit la valeur de `special`. Sinon une contrainte de string pourrait accepter `<think>`,
  dont chaque caractère est valide dans une string. `special` dit comment `decode` les traite,
  pas s'ils sont autorisés.

---

## 3. Tokens mixtes

- Tokens contenant `"` : **1 344**
- Tokens mélangeant `"` avec `,`, `}` ou `:` : **317 distincts**
  (348 en additionnant les lignes ci-dessous, car certains tokens contiennent deux de ces caractères)

| Caractère | Nb | Exemples (les plus courts) |
|---|---|---|
| `,` | 186 | `",` · `","` · `Ġ",` · `":",` |
| `}` | 88 | `"}` · `"},` · `"}}` · `}\"` |
| `:` | 74 | `":` · `":"` · `Ġ":` · `\":` |

À retenir : `"}}` = exactement la fin d'un appel ; `Ġ"` = espace + guillemet ; `\":` contient
un `\"` qui **ne ferme pas** la string.

---

## 4. Exercice : le JSON reste-t-il valide ?

Situation : le décodeur a écrit `"shrek` dans la valeur de `name`, **dernier** paramètre de
`fn_greet` → la suite attendue est `"}}`.

| Token | Texte obtenu | Valide ? | Pourquoi |
|---|---|---|---|
| `"` | `"shrek"` | ✅ | ferme la string ; début de `"}}` |
| `:"` | `"shrek:"` | ✅ syntaxe, ❌ valeur | `:` est **dans** la string, puis `"` ferme ; valeur `shrek:` |
| `",` | `"shrek",` | ❌ | ferme la string, mais `,` annonce un paramètre qui n'existe pas |
| `"}` | `"shrek"}` | ✅ | ferme la string ; début exact de `"}}` |
| `\"` | `"shrek\"` | ✅ | `"` échappé : la string reste **ouverte** (valeur `shrek"`) |
| `Ġ"` | `"shrek "` | ✅ syntaxe, ❌ valeur | l'espace est **dans** la string, puis `"` ferme |

**Vérification `json.loads` (`04_json_check.py`)** — fins qui rendent le JSON valide :

```
'"'   -> ['}}']
':"'  -> ['}}']
'",'  -> [' "x": 1}}']     ← valide en JSON, mais invente un paramètre : hors schéma
'"}'  -> ['}']
'\\"' -> ['"}}']           ← il faut encore un " : la string était bien ouverte
' "'  -> ['}}']
```

**JSON valide ≠ JSON conforme au schéma.** Le décodeur doit garantir les deux.

**Règle :** un token est accepté si, en lisant ses caractères **un par un** depuis l'état
courant (dans la string, après un `\`, string fermée…), aucun ne viole la grammaire. On n'exige
pas un JSON complet à l'instant t, seulement un **début valide** qui peut être complété aux pas
suivants.

---

## 5. Token spécial `<|im_start|>`

**`encode("<|im_start|>user")`** → 2 ids :

| Id | `decode([id])` |
|---|---|
| 151644 | `''` (token spécial, masqué par `decode`) |
| 872 | `'user'` |

**Contre-épreuve `encode("<|im_start>user")`** (sans le `|` final) → 6 ids :

| Id | `decode([id])` |
|---|---|
| 27 | `'<'` |
| 91 | `'\|'` |
| 318 | `'im'` |
| 4906 | `'_start'` |
| 29 | `'>'` |
| 872 | `'user'` |

151644 a disparu : la chaîne n'est plus reconnue comme token spécial, elle est découpée comme du
texte ordinaire par le BPE.

**Pourquoi c'est important :** le modèle connaît le format de conversation par l'id 151644, pas
par les caractères. Une balise mal écrite ne provoque **aucune erreur** : elle est découpée en
texte ordinaire, la structure n'est plus reconnue et la qualité baisse silencieusement.
**Parade :** définir les balises une seule fois, dans des constantes.

---

## 6. Découpage : `Ġ` et chiffres

Phrase : `"What is the sum of 265 and 345?"`

- **`Ġ`** marque un token **précédé d'un espace** (`Ġis`, `Ġthe`…). `What` n'en a pas (rien
  avant), `?` non plus (collé à `345`).
- **Nombres :** un token **par chiffre** (`2`, `6`, `5`).
- **Espace avant un nombre :** token **à part** (`Ġ`, id 220). Avant le BPE, le pré-découpage de
  Qwen isole chaque chiffre : l'espace ne peut pas s'y coller.

→ Phase 5 : la contrainte de nombre travaille chiffre par chiffre, et doit gérer l'espace (le
refuser, ou forcer `"a":` sans espace).

---

## 7. Génération non contrainte

Question : `Answer only with JSON like {"name": ..., "parameters": {...}}. What is the sum of 2 and 3?`
— greedy, 40 tokens.

### Expérience 1 — prompt brut, sans template

```
The correct answer is 5. But the question is phrased in a way that makes it seem like it's a trick question. So, what is the correct answer?

The correct answer
```

Pas de JSON : le modèle continue un texte au lieu de répondre, et se répète.

### Expérience 2 — chat template

```
<think>
Okay, the user is asking for the sum of 2 and 3. Let me think. The sum of two numbers is just adding them together. So 2 plus 3 equals
```

Il comprend qu'on lui pose une question, mais il « réfléchit » en prose et épuise les 40 tokens.

**Conclusion 1 + 2 :** le chat template améliore la **compréhension** de la tâche, mais le modèle
ne produit toujours pas la **forme** demandée, et le thinking consomme des tokens, donc du temps.

### Expérience 3 — chat template + `<think>\n\n</think>\n\n` vide

````
```json
{"name": "sum", "parameters": {"value": 5}}
```
Human: What is the sum of 2 and 3?
</think>I need to
````

**Avec arrêt sur `<|im_end|>` (151645) :** la sortie s'arrête juste après le bloc JSON. La prose
venait de la boucle, pas du modèle.

| Problème observé | Ce qui le résout |
|---|---|
| Markdown et prose autour du JSON | la structure `{"name": ...}` est **forcée** par le programme, pas générée |
| fonction inventée (`sum`) | le nom est choisi **uniquement** parmi les fonctions définies (`ChoiceConstraint`, phase 4) |
| mauvais paramètres / résultat calculé (`value: 5`) | les clés sont **forcées** depuis la définition, seules les valeurs sont générées (phase 6) |
| pas d'arrêt | le **programme** décide où le JSON se termine (`}}` forcé) ; `<|im_end|>` interdit pendant une valeur |

→ Matière directe pour « Challenges » du README : pourquoi le prompting seul ne suffit pas.

---

## 8. Temps d'un pas

Un appel à `get_logits_from_input_ids`, après warm-up, 5 mesures (CPU) :

| Prompt | Mesures (s) | Moyenne |
|---|---|---|
| 56 tokens | 0.346 · 0.279 · 0.294 · 0.263 · 0.288 | **0,29 s** |
| 441 tokens | 1.179 · 1.070 · 1.101 · 1.094 · 1.090 | **1,11 s** |

- Longueur × ~8 → temps × **3,8** : une partie du coût d'un appel est fixe.
- **Estimation :** 20 tokens générés avec un prompt de ~400 tokens = 20 appels × ~1,1 s
  ≈ **22 s par prompt**. `function_calling_tests.json` contient 11 prompts → **≈ 4 min** + le
  chargement. **Très proche de la limite de 5 min.**
- Leviers (phase 8) : segments forcés ajoutés **sans** appel au modèle ; prompt système court.

---

## 9. Récapitulatif « Terminé quand… »

| À savoir de mémoire | Réponse |
|---|---|
| Taille du vocab | 151 643 (+ 26 added tokens = 151 669 ; 14 `special`, 12 non) |
| Taille des logits | 151 936 |
| Temps de chargement (CPU, cache) | ~2,1 s |
| Durée d'un pas (CPU) | ~0,29 s (56 tokens) · ~1,11 s (441 tokens) |
| Forme de `encode` | tensor `[1, n]` → `[0].tolist()` → `list[int]` |
| Signification de `Ġ` | espace devant le token |
| Exemple de token mixte | `",` · `"}}` · `Ġ":` |

# Phase 2 — Modèles de données

## 10. Lire la structure de `functions_definition.json`

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  },
  "returns": {"type": "number"}
}
```

Il y a **deux niveaux** à ne pas mélanger :
- le niveau **fonction** (ce que décrit le fichier) : `name`, `description`, `parameters`, `returns` ;
- le niveau **appel** (ce que le LLM produira) : `a = 2`, `b = 3`. C'est là qu'apparaissent les `int`/`float`.

**Q1. Quels champs a une fonction, et quels sont leurs types Python ?**
- Ma réponse : ~~a et b, de type int ou float~~ → ça, ce sont les **paramètres** de *cette* fonction,
  et `int`/`float` sont les types des **valeurs au moment de l'appel** (phase 6+).
- ✅ Correction : une fonction a 4 champs :

| Champ | Exemple | Type Python après `json.load` |
|---|---|---|
| `name` | `"fn_add_numbers"` | `str` |
| `description` | `"Add two numbers…"` | `str` |
| `parameters` | `{"a": {"type": "number"}, …}` | `dict[str, dict]` |
| `returns` | `{"type": "number"}` | `dict` |

**Q2. `parameters` : liste ou dictionnaire ? Clés / valeurs ?**
- Ma réponse : dict, clés `a`/`b`, valeurs `number` → presque.
- ✅ Correction : c'est un **dict**. Les clés sont les **noms des paramètres** (`str`). Les valeurs ne sont **pas**
  `"number"`, mais un **objet** `{"type": "number"}`. Il y a un niveau d'imbrication de plus.
  En pydantic : `parameters: dict[str, ParameterSpec]`.
- Conséquence : un dict garde l'ordre d'insertion (Python ≥ 3.7). L'ordre `a` puis `b` est donc conservé, et
  ce sera l'ordre des clés forcées dans le JSON généré.

**Q3. `{"type": ...}` apparaît dans un paramètre ET dans `returns` : conclusion ?**
- Ma réponse : nécessaire pour ces 2 champs → c'est vrai, mais il faut aller plus loin.
- ✅ Correction : c'est la **même forme** aux deux endroits, donc **une seule classe** (`ParameterSpec`)
  sert pour les deux : `parameters: dict[str, ParameterSpec]` et `returns: ParameterSpec`.
  C'est le principe DRY (*Don't Repeat Yourself*) : si la forme évolue, on ne la change qu'à un endroit.

**Q4. Quels types dans le fichier ? Lesquels en correction ?**
- ✅ Dans le fichier : seulement `"number"` et `"string"` (vérifiable sur les 5 fonctions).
- Possibles en correction (architecture §5, « Les données d'entrée ») : `"integer"`, `"boolean"`.
  Ce sont les types de base de JSON Schema, la norme dont s'inspire ce format.
- Correspondance avec Python (utile en phase 6+) :

| `type` JSON | Valeur Python attendue | Exemple généré |
|---|---|---|
| `number` | `float` | `2.0`, `-3.5` |
| `integer` | `int` | `42` |
| `string` | `str` | `"shrek"` |
| `boolean` | `bool` | `true` / `false` |

---

## 11. Les modèles pydantic (`src/models.py`, testés dans `scratch/07_models.py`)

Lancer avec `uv run python -m scratch.07_models` depuis la racine. Avec `python scratch/07_models.py`,
Python cherche les imports dans `scratch/` et `from src.models import ...` lève `ModuleNotFoundError`.

### Les 4 classes

| Classe | Représente | Champs |
|---|---|---|
| `ParameterSpec` | `{"type": "number"}` | `type: ParamType` |
| `FunctionDefinition` | une fonction de `functions_definition.json` | `name`, `description`, `parameters: dict[str, ParameterSpec]`, `returns: ParameterSpec` |
| `PromptItem` | un élément de `function_calling_tests.json` | `prompt: str` |
| `FunctionCall` | une ligne du fichier de sortie | `prompt`, `name`, `parameters: dict[str, float \| int \| str \| bool]` |

- `ParamType = Literal["number", "string", "integer", "boolean"]` est un **alias de type** : il fixe une
  liste fermée de valeurs. `"banana"` est refusé dès le chargement (*fail fast*). On met les 4 types
  car la correction peut utiliser d'autres fichiers. Chaque type listé devra être générable en phase 5.
- **Le nom de l'attribut = la clé du JSON.** Avec `return_val` au lieu de `returns`, on obtient `Field required`.
- `prompt: str` est une **annotation** (un champ). `prompt = str` est une **affectation**, refusée par pydantic.
- `model_validate`, `model_dump`… viennent de `BaseModel` par **héritage**. `model_validate` est une
  méthode de **classe** : elle fabrique un objet à partir d'un dict.

### Compréhension de liste

```python
functions = []                                   # équivaut à :
for elem in raw_functions:                       # [FunctionDefinition.model_validate(elem)
    functions.append(FunctionDefinition.model_validate(elem))  #  for elem in raw_functions]
```
Ce qui était dans `append(...)` passe à gauche. Cette expression doit **utiliser `elem`**.

### Format des erreurs pydantic

| Cas testé | Chemin affiché | Type d'erreur |
|---|---|---|
| `"type": "banana"` (test 4) | `parameters.a.type` | `literal_error` (+ liste des valeurs permises) |
| `description` supprimée (test 5) | `description` | `missing` (`Field required`) |

- Pydantic liste **toutes** les erreurs d'un coup, avec le chemin complet dans la structure imbriquée.
- En phase 2, `io_utils.py` attrapera `ValidationError` et le convertira en `InputFileError`
  (nom du fichier + ce message), sans traceback.

### Ce que pydantic ne vérifie PAS

- `FunctionCall(parameters={"a": 2})` est accepté et **reste `int`** : pydantic en mode « smart » garde le
  type exact quand il est dans l'union. `"deux"` serait accepté aussi, car `str` est dans l'union.
- Raison : `FunctionCall` ne connaît pas la `FunctionDefinition` correspondante. Le lien
  « `a` est déclaré `number`, donc `float` » est **mon travail en phase 6** (conversion après génération).
- Piège pour la phase 6 : `isinstance(True, int)` vaut `True` (`bool` est une sous-classe d'`int`).
  Il faut tester `bool` **avant** `int`.
- Question de soutenance : « Pourquoi pydantic ne suffit-il pas à garantir les types de la sortie ? »

### Aliasing et `deepcopy`

```python
a = {"x": 1}
b = a            # pas une copie : un 2e nom pour le MÊME dict
b["x"] = 99      # → a vaut aussi {"x": 99}
```
- Observé : sans copie, le `"banana"` du test 4 réapparaissait dans le test 5.
- `copy.copy` ne copie que le **1er niveau** : les dicts imbriqués (`parameters`) restent partagés.
  `copy.deepcopy` copie **tous** les niveaux.
- Règle : faire la copie **avant de modifier**. Une copie faite après la modification copie les dégâts.

---

## 12. Erreurs et I/O (`src/errors.py`, `src/io_utils.py`) — rappel rapide

- Hiérarchie : `CallMeMaybeError` → `InputFileError` · `ModelError` (le **LLM**, pas pydantic) · `DecodingError`.
- **N'attraper que ce qu'on sait traiter.** `except CallMeMaybeError` = erreurs **prévues** (faute de
  l'utilisateur) → message propre. Un bug (`KeyError`…) doit garder son traceback → jamais `except Exception`.
- Traduction : on attrape l'erreur technique et on lève la nôtre, chaînée avec `raise InputFileError(f"{path}: {e}") from e`.
- `_read_json_list(path, what)` (privée, `_`) factorise : `open` + `json.load` → vérifie une **liste** → **non vide**.
  `load_functions` / `load_prompts` ne font plus que le `model_validate`.
- `write_results` : `model_dump()` sur chaque appel → `mkdir(parents=True, exist_ok=True)` → `json.dump(..., indent=2, ensure_ascii=False)`.
- `mkdir` : sans `parents` → `FileNotFoundError` ; 2ᵉ appel sans `exist_ok` → `FileExistsError`.
  Les deux héritent d'`OSError` → un seul `except OSError` couvre `mkdir` + `open`. Opération **idempotente**.
- `json.load(f)` lit (fichier → Python) ; `json.dump(obj, f)` écrit (Python → fichier, **objet d'abord**).
  `json.dumps` renvoie une chaîne. `json` ne sait pas écrire un objet pydantic → `model_dump()` avant.

---

## 13. La ligne de commande (`src/cli.py`)

**Rôle :** traduire les options tapées dans le terminal en un objet typé `CliConfig`.
Il ne connaît que des **chemins**. Il ne vérifie pas que les fichiers existent : c'est le travail d'`io_utils`.

| Entrée | Exemple | Module |
|---|---|---|
| options du terminal | `--input x.json` | `cli.py` → « **quels** fichiers ? » |
| contenu des fichiers | `[{"prompt": ...}]` | `io_utils.py` → « **que contiennent**-ils ? » |

Structure :
- 3 constantes en MAJUSCULES `DEFAULT_FUNCTIONS`, `DEFAULT_INPUT`, `DEFAULT_OUTPUT`, qui désignent des **fichiers**
  et non des dossiers (`open("data/output/", "w")` échoue).
- `CliConfig(BaseModel)` : 3 champs `Path`. Typé, contrairement à `args` dont tous les attributs sont `Any`.
- `parse_args(argv: list[str] | None = None) -> CliConfig` : un `add_argument` par option, avec `type=Path`, `default=`, `help=`.

Ce qu'argparse fait tout seul :
- **`--help` / `-h`** est généré à partir de `description` et des `help=`.
- **Une option inconnue ou une valeur invalide** affiche l'usage et l'erreur, puis quitte avec le **code 2**.
- **`type=`** : tout ce qui vient du terminal est du **texte**. `type=int` appelle `int("3")`, `type=Path` appelle `Path("x.json")`.
  On passe la classe **sans parenthèses** (`type=Path`, pas `Path()`) : c'est argparse qui l'appelle.
- **Sans `default=`**, une option absente vaut `None`. Bug rencontré : `CliConfig` refusait `None` au lieu de prendre le chemin par défaut.
  flake8 ne le voit pas, seule l'**exécution** le révèle.
- **`argv`** : `None` → argparse lit `sys.argv`. Une liste → on simule une ligne de commande, ce qui sert dans les tests.

---

## 14. Le point d'entrée (`src/__main__.py`)

`python -m src` exécute `src/__main__.py`. C'est la porte d'entrée : elle **enchaîne** les modules et gère les erreurs **à un seul endroit**.

```
main()
 ├─ parse_args()        → config
 ├─ load_functions()    → functions        lecture
 ├─ load_prompts()      → prompts          lecture
 ├─ _fake_calls()       → calls            traitement (faux pipeline, remplacé en phase 6)
 └─ write_results()                        écriture
```

- `_fake_calls` répond à chaque prompt avec la 1ʳᵉ fonction et `0.0` pour chaque paramètre.
  Compréhension de **dict** : `{name: 0.0 for name in fn.parameters}`. Boucler sur un dict donne ses **clés**.
  Piège évité : `fn.parameters` contient des `ParameterSpec` (des définitions), pas des valeurs.
- `main() -> int` **renvoie** un code au lieu de quitter elle-même, ce qui la rend testable. `sys.exit(main())` le transmet au système.
- `if __name__ == "__main__":` est vrai quand le fichier est **lancé**, pas quand il est **importé** (par un test, par exemple).
- Erreurs : stdout est réservé aux résultats, **stderr** aux diagnostics (`print(..., file=sys.stderr)`).

| Situation | Géré par | Code de sortie |
|---|---|---|
| succès | `return 0` dans le `try` | `0` |
| `CallMeMaybeError` | `print(f"Error: {e}", file=sys.stderr)` | `1` |
| mauvaise option | argparse | `2` |
| Ctrl+C (`KeyboardInterrupt`) | message `Interrupted` | `130` |
| bug imprévu | **non attrapé** → traceback | `1` (Python) |

Vérifié à la main : `make run` → 11 entrées ; `--input nope.json` → `Error: nope.json: [Errno 2] ...`, code 1, sans traceback.

Pièges rencontrés : `calls = load_prompts(config.output)` lisait le fichier de **sortie** au lieu de fabriquer les appels.
L'ordre des arguments de `write_results(path, calls)` : la signature fait foi.
F541 : une f-string sans `{}` ne doit pas avoir de `f`.

---

## 15. Les tests (`tests/`, lancer avec `uv run pytest -v`)

Configuration ajoutée dans `pyproject.toml`. Sans elle, `import src` échoue pendant les tests :
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Les bases de pytest :
- pytest exécute toute fonction `test_*` d'un fichier `tests/test_*.py`. On vérifie avec un simple `assert`.
- `with pytest.raises(Erreur, match="texte"):` réussit seulement si l'erreur est levée et que son message contient le texte.
- Les **fixtures** sont des paramètres au nom réservé, que pytest remplit tout seul :
  - `tmp_path` : un dossier temporaire vide, propre à chaque test, qui permet de créer de faux fichiers d'entrée ;
  - `monkeypatch` : remplace temporairement une valeur, par exemple `sys.argv` pour simuler une ligne de commande ;
  - `capsys` : capture stdout et stderr (`capsys.readouterr().err`).
- Un test vérifie **un** comportement, et son nom dit lequel. S'il échoue, on sait tout de suite ce qui est cassé.

| Fichier | Nb | Ce qui est vérifié |
|---|---|---|
| `test_io_utils.py` | 10 | vrais fichiers (5 / 11) · fichier absent · JSON cassé · pas une liste · liste vide · schéma invalide · `"banana"` (chemin `parameters.a.type`) · `write_results` crée les dossiers et garde `Zoé` · écriture impossible |
| `test_cli.py` | 3 | valeurs par défaut · une option remplace seulement son défaut · option inconnue → `SystemExit` code 2 |
| `test_models.py` | 4 | 4 types acceptés · `"banana"` refusé · ordre des paramètres conservé · `True` reste `bool`, `2` reste `int` |
| `test_main.py` | 2 | run complet → code 0 + 11 entrées `{prompt, name, parameters}` · fichier absent → code 1, `Error:` sur stderr, pas de traceback |

Résultat : **19 passed en 0,2 s.** Les fichiers `test_constraints.py`, `test_pipeline.py` et `test_vocabulary.py` sont vides, ils serviront aux phases 3 à 6.

**Pourquoi des tests, même s'ils ne sont pas notés :** après chaque modification, `uv run pytest` vérifie en 0,2 s
la dizaine de cas qu'on testait à la main. Ils détectent aussi des bugs que lint ne voit pas,
comme une exception sans héritage ou un `default=` oublié.

---

## 16. Récapitulatif phase 2 « Terminé quand… » ✅

| Critère | État |
|---|---|
| `make run` produit `data/output/function_calling_results.json` (faux contenu, vrai format) | ✅ 11 entrées |
| chaque cas d'erreur affiche un message clair, sans traceback | ✅ codes 1 et 2 |
| `make lint` | ✅ 0 erreur |
| `uv run pytest` | ✅ 19 tests |
