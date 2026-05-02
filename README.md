# NLP Task 2 — Earnings Call Boilerplate Classifier

---

## Project Structure

```
NLP Task2/
├── ECT/                        # 133 raw earnings call transcripts (.txt), 16 companies
├── data/
│   ├── sentence_pool.csv       # All unique sentences extracted from ECT/
│   └── gold_sample.csv         # Stratified 2,500-sentence sample used for labeling
├── labels/
│   ├── labels.csv              # 2,500 sentences + LLM judge predictions
│   ├── gold_labels_final.csv   # Final majority-vote gold standard
│   ├── features.csv            # 123-dim hand-crafted features for tree models
│   └── splits/
│       ├── train.csv           # 60% training set
│       ├── val.csv             # 20% validation set
│       └── test.csv            # 20% test set
├── models/
│   ├── lr_frozen_emb.pkl       # Logistic Regression on frozen embeddings
│   ├── fasttext_model.bin      # FastText supervised classifier
│   ├── rf_tree.pkl             # Random Forest (hand-crafted features)
│   ├── hgbm_tree.pkl           # HistGradientBoosting (hand-crafted features)
│   ├── finbert/                # Fine-tuned FinBERT (HuggingFace format)
│   └── *_meta.json             # Saved metrics & config for each model
├── checkpoints/
│   └── checkpoint-2500/        # SetFit training checkpoint
└── src/
    ├── 01_extract_sentences.py
    ├── 02–05_label_*.py        # LLM labeling (Qwen, Gemma, DeepSeek via Ollama)
    ├── 05_majority_vote.py
    ├── 00_make_splits.py
    ├── 06_linear_frozen_emb.py
    ├── 07_features.py
    ├── 08_tree_ensemble.py
    ├── 09_fasttext.py
    ├── 10_setfit.py
    ├── 11_finbert.py
    ├── 12_ensemble.py
    └── app.py                  # Streamlit inference GUI
```

---

## Environment

Python 3.10+. All pre-trained model files are already saved in `models/`.

```bash
pip install pandas numpy scikit-learn nltk sentence-transformers torch transformers fasttext-wheel streamlit
```

---

## Reproducing the Results

All model artifacts (`.pkl`, `.bin`, `finbert/` checkpoint) are pre-saved. No re-training is needed.

### 1. Evaluate the ensemble on the test set

Loads `models/lr_frozen_emb.pkl`, `models/fasttext_model.bin`, and `models/finbert/`, runs weighted prediction on `labels/splits/test.csv`, and prints the classification report.

```bash
cd "NLP Task2"
python src/12_ensemble.py
```

Expected output:

```
Ensemble: FinBERT 50% | LR 30% | FastText 20%
Best threshold : 0.13
recall_sub     : 0.9704
macro-F1       : 0.7539
```

### 2. Evaluate individual models

Each script below loads its own `.pkl` / `.bin` file and evaluates on the same test split.

```bash
python src/06_linear_frozen_emb.py   # LR on frozen all-MiniLM-L6-v2 embeddings
python src/08_tree_ensemble.py       # Random Forest + HistGradientBoosting
python src/09_fasttext.py            # FastText supervised classifier
python src/11_finbert.py             # Fine-tuned FinBERT
```

Saved metrics for every model are also in `models/*_meta.json`.

### 3. Run the interactive demo

```bash
streamlit run src/app.py
```

App will open in the browser at `http://localhost:8501`. On first load, the three models are downloaded/loaded into memory (takes ~30 seconds); subsequent runs are cached.

**Usage**:
- **Upload** a `.txt` earnings call transcript (samples available in `ECT/`), or **paste** any text directly into the text box
- Click **Classify** — the app splits the text into sentences and runs the ensemble on each one
- **Document view**: sentences highlighted green (substantive) or red (boilerplate)
- **Results table**: every sentence with its predicted label and `P(substantive)` score
- **Summary bar** at the top shows total sentence count and boilerplate/substantive breakdown

The threshold (0.13) and model weights are loaded automatically from `models/ensemble_meta.json` and require no manual configuration.

---

## Model Files

| File | Model |
|---|---|
| `models/lr_frozen_emb.pkl` | Logistic Regression on frozen `all-MiniLM-L6-v2` embeddings |
| `models/fasttext_model.bin` | FastText supervised classifier |
| `models/finbert/` | Fine-tuned FinBERT (HuggingFace format) |
| `models/rf_tree.pkl` | Random Forest on 123-dim hand-crafted features |
| `models/hgbm_tree.pkl` | HistGradientBoosting on same features |
| `models/ensemble_meta.json` | Ensemble weights, threshold, and test metrics |

