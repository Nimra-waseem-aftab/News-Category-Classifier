# News-Category-Classifier

A machine learning desktop application that classifies news articles into categories in real time. Paste any news text into a floating window and receive an instant category prediction with confidence scores, powered by Naive Bayes and Logistic Regression.

---

## Features

- Expression evaluation with correct operator precedence — Naive Bayes and Logistic Regression trained simultaneously, best model selected automatically
- Always-on-top floating desktop window for quick access without switching applications
- Clipboard integration — copy text from anywhere and classify it with a single keystroke
- Displays top 3 category predictions with individual confidence percentages
- Color-coded confidence levels: green (high), orange (medium), red (low)
- Model caching — trains once and loads from disk on all subsequent runs
- Comprehensive error handling — distinct messages for empty input, short text, and processing failures

---

## Error Handling

| Error | Cause |
|---|---|
| `Empty input` | User clicked Classify without entering any text |
| `Text too short` | Input is fewer than 10 characters |
| `No meaningful words` | Text contains only stopwords or very short tokens after preprocessing |
| `Classification error` | Internal model or vectorizer failure during prediction |

---

## Menu Options

```
NEWS CATEGORY CLASSIFIER
==============================

  Paste & Classify    Ctrl+V
  Classify            Enter
  Clear               Clears input and result
  Minimize            Esc
```

---

## Example Session

```
Input: Apple announces new MacBook with M4 chip at WWDC event

----------------------------
Predicted Category:  TECH
Confidence:          91.43%
----------------------------

Top 3 Predictions:
1. TECH          91.43%
2. SCIENCE        5.21%
3. BUSINESS       2.10%
```

---

## Requirements

| Requirement | Details |
|---|---|
| Language | Python 3.8+ |
| ML Library | scikit-learn (TF-IDF, Naive Bayes, Logistic Regression) |
| NLP | NLTK (WordNet lemmatizer, stopwords corpus) |
| GUI | tkinter (included with Python) |
| Clipboard | pyperclip |
| OS | Windows, macOS, Linux |
| Dataset | HuffPost News Category Dataset (JSON or CSV) |

---

## Installation & Running

### Standard Setup

```bash
git clone https://github.com/yourusername/news-classifier.git
cd news-classifier
pip install pandas numpy scikit-learn nltk pyperclip
python AI_Sem_project_code.py
```

### Linux Clipboard Support

```bash
sudo apt install xclip
pip install pandas numpy scikit-learn nltk pyperclip
python AI_Sem_project_code.py
```

---

## Project Structure

```
AI_Sem_project_code.py    <- Single-file source (all logic and GUI)
README.md

# Generated after first run:
model.pkl
vectorizer.pkl
preprocessor.pkl
model_info.pkl
```

### Key Components

| Component | Description |
|---|---|
| `main` | Entry point — loads dataset, trains or loads model, launches GUI |
| `TextPreprocessor` | Cleans text, removes stopwords, applies WordNet lemmatization |
| `clean_dataset` | Normalizes column names, removes invalid rows, filters categories |
| `load_dataset` | Auto-detects and loads JSON or CSV dataset files |
| `train_models` | Trains Naive Bayes and Logistic Regression, saves the better model |
| `NewsClassifierPopup` | tkinter GUI — always-on-top floating window with input and result display |
| `classify_text` | Preprocesses input, vectorizes, runs prediction, extracts top 3 results |
| `paste_and_classify` | Reads clipboard via pyperclip and triggers classification immediately |
| `display_result` | Renders prediction, confidence bar, and top 3 alternatives in the GUI |

---

## Algorithm

The classifier uses a classic **TF-IDF + linear model** pipeline:

1. **Load** the dataset and normalize all column names to a standard schema.
2. **Filter** categories — keep only those with at least 10 samples; retain top 20 by frequency.
3. **Preprocess** each article — strip URLs and special characters, lowercase, lemmatize, remove stopwords.
4. **Vectorize** using TF-IDF with 2,000 features and a unigram/bigram range.
5. **Train** both Multinomial Naive Bayes and Logistic Regression on an 80/20 split.
6. **Select** the model with the higher test accuracy and serialize all artifacts to disk.
7. **Predict** — at runtime, new text passes through the same preprocessing and vectorization before inference.

## License

This project is provided for educational purposes as part of an AI semester project. Feel free to use and modify it for learning machine learning and NLP concepts.
