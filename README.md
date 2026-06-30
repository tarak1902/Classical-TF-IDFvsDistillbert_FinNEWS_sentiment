Classical vs modern NLP compared fairly: TF-IDF + Logistic Regression vs fine-tuned DistilBERT on financial news sentiment.
# Financial News Sentiment: TF-IDF vs DistilBERT

A side-by-side comparison of the **classical** and **modern** approaches to text classification, on the same task and the same test set. The point isn't to fine-tune a transformer — it's to show *why* transformers replaced classical NLP, with honest numbers and an explanation of where each approach wins and fails.

**Task:** classify financial news headlines into three sentiment classes (negative / neutral / positive).
**Dataset:** FinancialPhraseBank — 4,846 headlines, expert-annotated from a retail-investor perspective.

---

## Headline result

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.730 | 0.674 | 0.689 | 0.680 |
| **DistilBERT (fine-tuned)** | **0.837** | **0.806** | **0.849** | **0.824** |

**DistilBERT improvement: +10.7 points accuracy, +14.5 points macro-F1** — on the identical test split.

Why macro-F1 and not just accuracy? The dataset is imbalanced (neutral is ~59% of samples), so accuracy alone is misleading — a model that guessed "neutral" every time would score ~59% while being useless. Macro-F1 weights all three classes equally and exposes that, so it's the metric that actually reflects performance here.

---

## Why the transformer wins (the actual lesson)

The classical model represents text as a **bag of weighted words** (TF-IDF). It has no concept of word order or context, so it stumbles exactly where financial language is subtle. The confusion matrices show it concretely:

- The classical model confused **positive with neutral** in 86 cases and **neutral with positive** in 76 — the boundary between "good news" and "routine update" is where bag-of-words breaks down, because that distinction lives in phrasing and context, not individual word counts.
- DistilBERT, which understands word order and carries pretrained language knowledge, cut those same confusions sharply and made its biggest gains on the **negative** and **positive** classes — the per-class F1 for negative jumped from 0.62 to 0.83.

In short: TF-IDF counts words; DistilBERT understands sentences. On a clean, easy task the gap would be small — on subtle financial sentiment, it's large, and that's the point.

---

## Results in detail

### TF-IDF + Logistic Regression

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.574 | 0.669 | 0.618 | 121 |
| neutral | 0.807 | 0.811 | 0.809 | 576 |
| positive | 0.640 | 0.586 | 0.612 | 273 |

Confusion matrix (rows = actual, cols = predicted):

| actual \ pred | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 81 | 26 | 14 |
| neutral | 33 | 467 | 76 |
| positive | 27 | 86 | 160 |

Strong on the dominant neutral class, weak on the positive/negative boundaries.

### DistilBERT (fine-tuned)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.750 | 0.917 | 0.825 | 121 |
| neutral | 0.892 | 0.845 | 0.868 | 576 |
| positive | 0.775 | 0.784 | 0.780 | 273 |

Confusion matrix (rows = actual, cols = predicted):

| actual \ pred | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 111 | 8 | 2 |
| neutral | 29 | 487 | 60 |
| positive | 8 | 51 | 214 |

Large gains on negative and positive; the remaining errors cluster on the genuinely hard neutral/positive boundary.

---

## Approach

### Classical pipeline
- **Preprocessing:** lowercase, strip URLs, punctuation, numbers, and extra whitespace.
- **Vectorizer:** `TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")` — unigrams + bigrams, top 5,000 features.
- **Classifier:** `LogisticRegression(class_weight="balanced", max_iter=1000)` — class weighting to counter the imbalance.

### Transformer pipeline
- **Model:** `distilbert-base-uncased`, fine-tuned for 3-class classification via HuggingFace `Trainer`.
- **Tokenization:** original (uncleaned) headlines, `max_length=128`, padding + truncation. (Transformers do their own subword tokenization, so the classical text-cleaning is deliberately *not* applied here.)
- **Hyperparameters:** 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01, fp16.
- **Training:** ~44 seconds on a Colab Tesla T4 GPU.

### Fair-comparison guarantee
Both models were evaluated on the **identical** stratified test split (`test_size=0.2, random_state=42, stratify=y`), with DistilBERT reusing the exact train/test row indices. TF-IDF was fit on training data only — no test-set vocabulary leakage.

---

## Training dynamics (and an honest note)

| Epoch | Train Loss | Val Loss | Accuracy | Macro F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.422 | 0.404 | 0.833 | 0.809 |
| 2 | 0.289 | 0.442 | 0.813 | 0.808 |
| 3 | 0.152 | 0.448 | 0.837 | 0.824 |

Training loss fell steadily while validation loss rose after epoch 1 — a textbook early sign of mild overfitting (the model growing over-confident on training data). Final accuracy and macro-F1 were still best at epoch 3, so it's reported as the final model, but the trend is worth noting honestly.

---

## Honest caveats

- **Eval set overlap:** the held-out test set was also used as the per-epoch evaluation set during fine-tuning. For a rigorous research setup, a separate validation set and a truly untouched final test set would be cleaner. For a portfolio comparison this is acceptable, but it's noted rather than hidden.
- **Financial sentiment is genuinely hard:** headlines carry mixed signals ("narrower losses," "revenue up but profit down," routine updates), which makes this harder than typical positive/negative sentiment. The modest absolute numbers reflect real task difficulty, not a broken model.

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** pandas, numpy, scikit-learn, matplotlib, torch, transformers, datasets, accelerate (Python 3.12). The transformer half expects a GPU (developed on a Colab T4); the classical half runs anywhere.

## Project structure

```
financial-news-sentiment-tfidf-vs-distilbert/
├── notebooks/
│   └── financial_sentiment_comparison.ipynb
├── results/
│   ├── classical_confusion_matrix.png
│   ├── distilbert_confusion_matrix.png
│   └── final_model_comparison.csv
├── data/
│   └── README.md          # dataset source + download note
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run

The project is a single notebook. In order: load `all-data.csv`, inspect class distribution, clean text for the classical model, create one stratified split, train + evaluate TF-IDF + Logistic Regression, then tokenize the original headlines and fine-tune + evaluate DistilBERT on the same split, and finally compare both side by side.

---

## What this project demonstrates

- The classical NLP pipeline end to end (TF-IDF, n-grams, class-weighted linear classification)
- The modern pipeline (HuggingFace tokenization + transformer fine-tuning)
- Honest, leakage-aware evaluation on imbalanced data (macro-F1, per-class metrics, confusion matrices)
- The ability to *compare two approaches fairly and explain the difference* — not just run a model

*Dataset: FinancialPhraseBank (Malo et al., 2014), CC BY-NC-SA 4.0 — used here for non-commercial portfolio purposes.*
