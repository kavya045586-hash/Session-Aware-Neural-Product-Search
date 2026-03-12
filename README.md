# Session Recommender — Complete Setup Guide

## All Files Status

| File | Status | Purpose |
|---|---|---|
| step1_merge.py | Correct | Parse raw data + extract bought_together |
| step2_clean.py | Correct | Clean data + inject co-purchase training rows |
| step3_vectorize.py | Correct | Vectorize item titles (with resume support) |
| step4_train.py | Correct | Train two-tower model with InfoNCE loss |
| step5_evaluate.py | Correct | Evaluate model + sanity check |
| app.py | Correct | Flask API with smart search |
| index.html | Correct | Amazon-style frontend |

---

## Virtual Environment Setup (ONE TIME ONLY)

```bash
# Step 1 - Create virtual environment
python -m venv venv

# Step 2 - Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Step 3 - Install all packages
pip install -r requirements.txt
```

---

## requirements.txt contents

```
pandas==2.2.2
numpy==1.26.4
pyarrow==16.1.0
tensorflow==2.16.2
sentence-transformers==3.0.1
flask==3.0.3
flask-cors==4.0.1
scikit-learn==1.5.1
tqdm==4.66.4
```

---

## Every Time You Open a New Terminal

```bash
# Just activate - no reinstall needed
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

---

## Required Folder Structure

```
your_project/
    venv/                          <- virtual environment (created once)
    data/
        Electronics.jsonl.gz       <- Amazon dataset
        meta_Electronics.jsonl.gz  <- Amazon dataset
        processed/                 <- auto-created by scripts
    models/                        <- auto-created by step4
    results/                       <- auto-created by step5
    step1_merge.py
    step2_clean.py
    step3_vectorize.py
    step4_train.py
    step5_evaluate.py
    app.py
    index.html
    requirements.txt
```

---

## Run Order (First Time Only)

```bash
python step1_merge.py                      # ~20-40 min
python step2_clean.py                      # ~5 min
python step3_vectorize.py                  # ~3-5 hrs CPU  /  ~20 min GPU
python step4_train.py --model session      # ~1-2 hrs CPU  /  ~20 min GPU
python step5_evaluate.py --model session   # ~5 min
python app.py                              # open http://127.0.0.1:5000
```

If step3 gets interrupted, just re-run it - it resumes from checkpoint automatically.

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| ModuleNotFoundError | pip install -r requirements.txt |
| model_session.keras not found | run step4 first |
| item_catalog.parquet not found | run step2 first |
| item_text_vectors.npy not found | run step3 first |
| copurchase_pairs.parquet not found | run step1 first |
| Port 5000 already in use | kill old process or change port in app.py |
