# MindScan — Project Structure & File Reference

**NCI H9DAI Research Project 2026**  
For team onboarding. Explains every folder and file, why it exists, and how it connects to the rest of the system.

---

## Quick Overview

MindScan is a Flask web app that runs mental health text through **12 models** simultaneously — 4 models across 3 independent datasets — and returns predictions with confidence scores.

```
User types text
      │
   app.py  (Flask server — receives HTTP request)
      │
  predict.py  (all prediction logic)
      │
  ┌───┴────────────────────────┐
  │                            │
Classical models          Transformer models
(models/classical/)       (models/transformers/)
  │                            │
tfidf → LR / SVM / XGBoost    XLM-RoBERTa
  │                            │
  └───────────┬────────────────┘
              │
         JSON response → UI (templates/index.html)
```

---

## Root-Level Files

### `app.py`
The Flask web server. Entry point — run this to start the app.

**What it does:**
1. At startup, calls `load_all_models()` from `predict.py` — loads all 12 models into RAM once.
2. Serves `templates/index.html` at `GET /`
3. Accepts user text at `POST /predict`, passes it to `predict_all()`, returns JSON.
4. Exposes `GET /health` to check if models are ready.

**Key code:**
```python
# Loads everything once on startup — not per request
load_all_models()

@app.route('/predict', methods=['POST'])
def predict():
    result = predict_all(text)   # runs all 12 models
    return jsonify(result)
```

**Validations it enforces before prediction:**
- `text` field must exist in the request body
- Text cannot be empty
- Text cannot exceed 5000 characters
- Models must be loaded (returns 503 if startup hasn't finished)

---

### `predict.py`
The brain of the system. Contains all model loading and prediction logic. Never run this directly — `app.py` imports it.

**Four functions:**

| Function | Purpose |
|----------|---------|
| `load_all_models()` | Called once at startup. Loads all PKL files + XLM-RoBERTa into `_models{}` dict |
| `clean_text(text)` | Strips URLs, @mentions, hashtags, punctuation, extra spaces. Applied before classical models |
| `predict_classical(text_clean, ds)` | Runs cleaned text through TF-IDF → LR/SVM/XGBoost for one dataset |
| `predict_transformer(text_raw, ds)` | Runs raw text through XLM-RoBERTa tokenizer → model for one dataset |
| `predict_all(raw_text)` | Calls all of the above across all 3 datasets, assembles final JSON |

**Why `clean_text` only goes to classical models:**
Classical models use TF-IDF (bag of words) — punctuation and case create noise.
XLM-RoBERTa's tokenizer handles this natively, so raw text gives better results.

**The `_models` dictionary — what gets stored:**
```python
_models = {
    # Preprocessors (one per dataset)
    'tfidf_d1': <TfidfVectorizer>,  'le_d1': <LabelEncoder>,
    'tfidf_d2': <TfidfVectorizer>,  'le_d2': <LabelEncoder>,
    'tfidf_d3': <TfidfVectorizer>,  'le_d3': <LabelEncoder>,

    # Classical classifiers (3 models × 3 datasets = 9)
    'logistic_regression_d1': <model>, 'svm_d1': <model>, 'xgboost_d1': <model>,
    'logistic_regression_d2': <model>, 'svm_d2': <model>, 'xgboost_d2': <model>,
    'logistic_regression_d3': <model>, 'svm_d3': <model>, 'xgboost_d3': <model>,

    # Transformers (1 tokenizer shared, 3 models)
    'tokenizer':   <XLMRobertaTokenizer>,
    'xlmr_d1':     <XLMRobertaForSequenceClassification>,  'xlmr_d1_len': 128,
    'xlmr_d2':     <XLMRobertaForSequenceClassification>,  'xlmr_d2_len': 128,
    'xlmr_d3':     <XLMRobertaForSequenceClassification>,  'xlmr_d3_len': 256,
    'device':      'cpu' or 'cuda',
}
```

**Suicide risk flag logic:**
```python
# D3 majority vote — if 3 or more of 4 models say "suicide", flag fires
suicide_count = sum(1 for r in d3.values()
    if 'suicide' in r['label'].lower() and 'non' not in r['label'].lower())
risk_flag = suicide_count >= 3
```

---

### `requirements.txt`
Exact library versions the project was built and tested with.

```
flask==3.0.3        → web server
scikit-learn==1.6.1 → LR, SVM, TF-IDF, LabelEncoder
xgboost==2.0.3      → XGBoost classifier
transformers==4.41.2→ AutoTokenizer, AutoModelForSequenceClassification
torch==2.3.0        → tensor ops + GPU support for XLM-RoBERTa
joblib==1.4.2       → loading .pkl files
numpy==1.26.4       → array operations (softmax, argmax)
```

**Important:** Do not upgrade these without retesting. `scikit-learn` version mismatches can corrupt PKL file loading silently.

---

## `Dataset/` Folder

Raw CSV files used to train all models. Not used by the running app — only by the training notebooks.

| File | Dataset | Task | Source |
|------|---------|------|--------|
| `DA_1_DB.csv` | D1 | Depression type — 6 classes | Zenodo (Nusrat 2024) |
| `DA_DB_2.csv` | D2 | Binary depression (0 = no, 1 = yes) | Kaggle (albertobellardini) |
| `DA_DB_3.csv` | D3 | Suicide risk — full dataset | Kaggle (nikhileswarkomati) |
| `DA_DB_3_H1.csv` | D3 split | First half of D3 — used in split study | Derived from DA_DB_3.csv |
| `DA_DB_3_H2.csv` | D3 split | Second half of D3 — used in split study | Derived from DA_DB_3.csv |

**D1 class labels (6 classes):**
Anxiety, Bipolar, Depression, Normal, PersonalityDisorder, PTSD

**D2 class labels (binary):**
0 = Not Depressed, 1 = Depressed
(The app maps these to readable strings via `D2_LABEL_MAP` in `predict.py`)

**D3 class labels (binary):**
Suicide, Non-Suicidal

---

## `notebooks/` Folder

Jupyter notebooks where all model training happened. These are offline tools — they produced the `.pkl` and `.safetensors` files that the app loads.

### `DA_Notebook_One.ipynb`
Trains all models for **Dataset 1** (D1 — 6-class depression type).

What it does in order:
1. Loads `DA_1_DB.csv`
2. Cleans text (same `clean_text` logic as `predict.py`)
3. Fits `TfidfVectorizer` on training data → saves `tfidf_d1.pkl`
4. Fits `LabelEncoder` on labels → saves `le_d1.pkl`
5. Trains LR, SVM, XGBoost, Random Forest → saves each as `_d1.pkl`
6. Fine-tunes XLM-RoBERTa base on D1 → saves `xlmr_d1_final/`
7. Generates confusion matrices (`cm_D1_*.png`) and EDA chart (`eda_d1.png`)
8. Writes results to `classical_results.csv`

### `DA_2_Notebook.ipynb`
Same pipeline for **Dataset 2** (D2 — binary depression). Produces all `_d2` files.

### `DA_3_SplitStudy.ipynb`
Trains models for **Dataset 3** (D3 — suicide risk) with an additional split study:
trains on `DA_DB_3_H1.csv`, validates on `DA_DB_3_H2.csv` to test generalisation.
Produces all `_d3` files and confusion matrices.

---

## `models/classical/` Folder

All classical ML model files. Every file is a frozen Python object saved with `joblib`.

### Naming convention
```
{model_type}_{dataset}.pkl
```
`d1` = Dataset 1, `d2` = Dataset 2, `d3` = Dataset 3.

---

### `tfidf_d1.pkl` / `tfidf_d2.pkl` / `tfidf_d3.pkl`

**What it is:** A fitted `TfidfVectorizer` (scikit-learn).

**What TF-IDF does:**
Converts a text string into a numeric vector. Each position in the vector = a word from the training vocabulary. The value = how important that word is in this document.

```
TF  = how often the word appears in THIS text
IDF = log(total documents / documents containing this word)
      → rare words score higher, common words ("the", "is") score near 0

score = TF × IDF
```

**Why it must be saved as `.pkl`:**
During training, TF-IDF assigns each word a fixed column index:
```
"hopeless"  → column 4821
"depressed" → column 1203
"therapy"   → column 9045
```
The classifier is trained expecting exactly this column layout.
If you re-fit TF-IDF at inference time, the columns shuffle:
```
"hopeless"  → column 6012   ← WRONG — model reads wrong feature
```
Saving the fitted object freezes the vocabulary and column order permanently.

**How it's used in the app:**
```python
# predict.py:60-61 (load)
_models['tfidf_d1'] = joblib.load('models/classical/tfidf_d1.pkl')

# predict.py:126 (inference)
vec = tfidf.transform([text_clean])   # text → sparse numeric matrix
```

---

### `le_d1.pkl` / `le_d2.pkl` / `le_d3.pkl`

**What it is:** A fitted `LabelEncoder` (scikit-learn).

**What LabelEncoder does:**
Creates a frozen mapping between string class names and integers.

D1 example (sorted alphabetically at training time):
```
"Anxiety"              → 0
"Bipolar"              → 1
"Depression"           → 2
"Normal"               → 3
"PersonalityDisorder"  → 4
"PTSD"                 → 5
```

**Why it must be saved as `.pkl`:**
Models output integers (`model.predict()` returns `2`).
The `le.classes_` array translates `2` back to `"Depression"`.
If the ordering was ever different during training, `2` could have meant `"Normal"`.
The saved file guarantees the mapping is identical to what the model learned.

**How it's used in the app:**
```python
# predict.py:60 (load)
_models['le_d1'] = joblib.load('models/classical/le_d1.pkl')

# predict.py:137-138 (inference)
pred_idx  = model.predict(vec)[0]    # model returns e.g. integer 2
raw_label = le.classes_[pred_idx]    # le.classes_[2] → "Depression"
```

**Note:** LabelEncoder is also used by XLM-RoBERTa (`predict.py:197`) — the transformer also outputs an integer index, and the same `le` decodes it.

---

### Classifier files

Each is a fully trained scikit-learn or XGBoost model saved with `joblib`.

| File pattern | Algorithm | Notes |
|-------------|-----------|-------|
| `logistic_regression_d*.pkl` | Logistic Regression | Linear model. Fast. Has `predict_proba` — gives clean probability output |
| `svm_d*.pkl` | Support Vector Machine | Max-margin classifier. Uses `decision_function` for confidence (converted to probability via softmax in `predict.py:150-154`) |
| `xgboost_d*.pkl` | XGBoost | Gradient-boosted trees. Has `predict_proba`. Usually strongest classical performer |
| `random_forest_d*.pkl` | Random Forest | Saved but **not loaded by the app**. Excluded — 646 MB file size, worst accuracy on D1/D3 |

**How classifiers are used:**
```python
# predict.py:65-70 (load — all 9 models in a loop)
for model_name in ['logistic_regression', 'svm', 'xgboost']:
    for ds in ['d1', 'd2', 'd3']:
        _models[f'{model_name}_{ds}'] = joblib.load(f'{model_name}_{ds}.pkl')

# predict.py:136-137 (inference)
model    = _models['svm_d1']
pred_idx = model.predict(vec)[0]   # vec came from tfidf.transform()
```

---

### Evaluation images & results

| File | What it shows |
|------|--------------|
| `cm_D1_SVM.png` | Confusion matrix for SVM on D1 — rows = actual, cols = predicted |
| `cm_D2_XGBoost.png` | Same format for XGBoost on D2 |
| `cm_D3_*.png` | All 4 confusion matrices for D3 |
| `eda_d1.png` / `eda_d2.png` / `eda_d3.png` | Class distribution bar charts per dataset |
| `classical_comparison.png` | Side-by-side accuracy chart across all models and datasets |
| `classical_results.csv` | Full metrics table (accuracy, F1, precision, recall) for every model |

These are generated by the notebooks and stored here for reference. Not used at runtime.

---

## `models/transformers/` Folder

Contains three subfolders, one per fine-tuned XLM-RoBERTa model.

**XLM-RoBERTa** = Cross-lingual Language Model RoBERTa.
Pre-trained by Meta on 100 languages and 2.5TB of text.
Your notebooks fine-tuned it further on each dataset's specific mental health text.

---

### `xlmr_d1_final/` — 6-class depression type model

#### `tokenizer_config.json`
Loaded first when `AutoTokenizer.from_pretrained()` is called.
Tells the loader which tokenizer class to use and defines special tokens.

```json
{
  "tokenizer_class": "XLMRobertaTokenizer",
  "bos_token":  "<s>",     ← placed at START of every input (classification signal)
  "eos_token":  "</s>",    ← placed at END of every input (sentence boundary)
  "pad_token":  "<pad>",   ← fills empty positions when text < max_length
  "mask_token": "<mask>",  ← used during pre-training only, not in your app
  "unk_token":  "<unk>",   ← replaces words not in vocabulary
  "model_max_length": 512
}
```

#### `tokenizer.json`
The actual vocabulary and tokenization rules. Loaded alongside `tokenizer_config.json`.

**Contains:**
- 250,002 subword tokens and their IDs
- Truncation rules: cut at 128 tokens from the right if text is longer
- Padding rules: pad to fixed length 128 with `<pad>` (ID=1) on the right

XLM-RoBERTa uses **SentencePiece subword tokenization** — it breaks words into pieces:
```
"hopelessness" → ["▁hope", "less", "ness"] → token IDs [2101, 394, 7823]
```
The `▁` symbol means "starts a new word". This approach handles misspellings, rare words, and any language because unknown words always decompose into known subpieces.

**Why only one tokenizer is loaded for all 3 XLM-R models:**
```python
# predict.py:82-83
tokenizer_path = os.path.join(TRANSFORMER_DIR, 'xlmr_d1_final')
_models['tokenizer'] = AutoTokenizer.from_pretrained(tokenizer_path)
```
All three models were fine-tuned from the same base checkpoint — their vocabularies are identical. Loading one saves ~50MB RAM and startup time.

#### `config.json`
Loaded when `AutoModelForSequenceClassification.from_pretrained()` is called.
Tells PyTorch how to construct the model architecture before loading weights.

Key fields:
```json
{
  "architectures": ["XLMRobertaForSequenceClassification"],
  "hidden_size": 768,           ← each token becomes a 768-number vector
  "num_hidden_layers": 12,      ← 12 transformer blocks stacked
  "num_attention_heads": 12,    ← 12 parallel attention patterns per layer
  "max_position_embeddings": 514, ← supports up to 512 tokens + 2 special tokens
  "vocab_size": 250002,         ← must match tokenizer.json exactly
  "id2label": {                 ← placeholder labels (app uses le.classes_ instead)
    "0": "LABEL_0", ... "5": "LABEL_5"
  }
}
```

The output layer has **6 neurons** because D1 has 6 classes. This is the only structural difference between the three XLM-R model folders.

#### `model.safetensors`
The actual trained weights — approximately 125 million floating-point numbers.
This file is the result of fine-tuning the base XLM-R checkpoint on your D1 training data.

**Why `.safetensors` and not `.pkl`:**
- `.pkl` can execute arbitrary Python code when loaded — a security risk
- `.safetensors` is a pure tensor format — no code execution, faster to load, can be memory-mapped directly

---

### `xlmr_d2_final/` — Binary depression model

Identical file structure to `xlmr_d1_final/`.

**Key difference in `config.json`:**
No `id2label` entries beyond 2 — this model's output layer has **2 neurons** (binary classification).

**Key difference in `tokenizer.json`:**
`max_length` during inference is 128 (same as D1). Set in `predict.py:86`:
```python
for ds, max_len in [('d1', 128), ('d2', 128), ('d3', 256)]:
```

---

### `xlmr_d3_final/` — Suicide risk model

Identical file structure. Output layer has **2 neurons** (Suicide / Non-Suicidal).

**Key difference:** `max_length = 256` at inference. Suicide-related posts tend to be longer — the model was fine-tuned with a longer context window to capture this.

---

## `templates/` Folder

### `templates/index.html`
The entire frontend — a single HTML file with embedded CSS and JavaScript.
Served by Flask at `GET /` via `render_template('index.html')`.

**What it does:**
- Provides the text input form
- Sends `POST /predict` with the user's text as JSON
- Receives the prediction JSON and renders results per dataset
- Shows the probability breakdown bars (using `class_probs` from XLM-R D1)
- Displays the suicide risk flag if `risk_flag: true`

---

## Full Request-to-Response Code Flow

```
1. User opens browser → GET /
   app.py:41-44    → render_template('index.html')

2. User types text and clicks Analyse → POST /predict { "text": "..." }
   app.py:47-77    → validates input, calls predict_all(text)

3. predict.py:221  → predict_all(raw_text) starts

4. predict.py:106  → clean_text(raw_text)
                      strips URLs, @mentions, punctuation → text_clean

5. predict.py:119  → predict_classical(text_clean, 'd1')
   predict.py:126      tfidf_d1.transform([text_clean]) → sparse vector
   predict.py:136      for each of LR, SVM, XGBoost:
   predict.py:137        model.predict(vec) → integer index
   predict.py:138        le_d1.classes_[index] → class name string
   predict.py:147        model.predict_proba() or decision_function() → confidence
                      returns { 'Logistic Regression': {label, confidence}, ... }

6. predict.py:166  → predict_transformer(raw_text, 'd1')
   predict.py:184      tokenizer(text, max_length=128) → token IDs tensor
   predict.py:192      xlmr_d1(token_ids) → logits (raw scores)
   predict.py:195      softmax(logits) → probabilities [0.04, 0.09, 0.72, ...]
   predict.py:196      argmax → predicted index
   predict.py:197      le_d1.classes_[index] → class name string
                      returns { label, confidence, all_probs }

7. Steps 5-6 repeat for 'd2' and 'd3'

8. predict.py:266  → suicide vote count across 4 D3 models
                      risk_flag = True if ≥3/4 models say "suicide"

9. predict.py:247  → winner per dataset = model with highest confidence

10. predict_all returns full JSON dict
    app.py:73-74   → jsonify(result) → HTTP 200 response

11. index.html JS  → receives JSON, renders results in the UI
```

---

## Dataset–Model–File Mapping

| Dataset | Task | TF-IDF | LabelEncoder | Classical Models | Transformer |
|---------|------|--------|-------------|-----------------|-------------|
| D1 | 6-class depression type | `tfidf_d1.pkl` | `le_d1.pkl` | `lr_d1`, `svm_d1`, `xgb_d1` | `xlmr_d1_final/` |
| D2 | Binary depression | `tfidf_d2.pkl` | `le_d2.pkl` | `lr_d2`, `svm_d2`, `xgb_d2` | `xlmr_d2_final/` |
| D3 | Suicide risk | `tfidf_d3.pkl` | `le_d3.pkl` | `lr_d3`, `svm_d3`, `xgb_d3` | `xlmr_d3_final/` |

Each row is completely independent — mixing files across rows (e.g. `tfidf_d1` with `lr_d2`) would silently produce wrong predictions with no error.

---

## Why PKL Files Must Never Be Regenerated Without Retraining

If you re-fit `tfidf_d1` on different data and save a new `tfidf_d1.pkl`:
- The column layout changes
- The existing `logistic_regression_d1.pkl` model was trained on the old layout
- Predictions become meaningless — no error is thrown, outputs are just wrong

The PKL files are a **matched set**. `tfidf_d1`, `le_d1`, `lr_d1`, `svm_d1`, `xgb_d1` were all produced in the same notebook run and must stay together. If any one is regenerated, all five must be regenerated together from the same training run.
