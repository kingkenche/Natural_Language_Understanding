# 🏈📰 Sport vs Politics — Text Classifier

> **NLU Assignment-1, Problem 4** — Binary text classification comparing three ML techniques across three feature representations.
>
> 🔗 **GitHub:** [https://github.com/kingkenche/Natural_Language_Understanding](https://github.com/kingkenche/Natural_Language_Understanding)

---

## 📌 Overview

This project builds a machine learning pipeline that classifies BBC news articles as **Sport** or **Politics**. We compare **9 experiments** (3 classifiers × 3 feature sets) and provide a detailed report.

### Results at a Glance

| Classifier           | BoW       | TF-IDF    | Bigram TF-IDF |
|----------------------|-----------|-----------|---------------|
| Logistic Regression  | 98.92 %   | 99.46 %   | 99.46 %       |
| Multinomial NB       | **100 %** | **100 %** | **100 %**     |
| Linear SVM           | 98.92 %   | 99.46 %   | 99.46 %       |

🏆 **Best model:** Multinomial Naïve Bayes — 100 % accuracy on all feature sets.

---

## 📁 Project Structure

```
Assignment-1/
├── data/                      # BBC News dataset (auto-downloaded)
├── notebooks/
│   └── eda.py                 # Exploratory Data Analysis
├── src/
│   ├── data_loader.py         # Download, clean & split data
│   ├── features.py            # BoW, TF-IDF, Bigram TF-IDF
│   ├── train.py               # Train 9 experiments, save results
│   └── evaluate.py            # Generate comparison charts
├── outputs/
│   ├── figures/               # All generated plots
│   └── results/               # JSON results
├── report/
│   └── report.md              # Full 5+ page report
├── requirements.txt
└── README.md                  # ← You are here
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA (auto-downloads dataset)

```bash
python notebooks/eda.py
```

### 3. Train All Models

```bash
python src/train.py
```

### 4. Generate Evaluation Charts

```bash
python src/evaluate.py
```

---

## 📊 Dataset

- **Source:** [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html) (Greene & Cunningham, 2006)
- **Classes:** Sport (511 docs), Politics (417 docs) — **928 total**
- **Split:** 80 % train (742) / 20 % test (186), stratified

### Preprocessing

1. Lowercase → remove digits & punctuation
2. Remove English stop words (NLTK)
3. WordNet lemmatisation

---

## 🔬 Feature Representations

| Method         | Description                          | Dimensions |
|----------------|--------------------------------------|------------|
| **BoW**        | Raw token counts                     | ≤ 5 000    |
| **TF-IDF**     | Term frequency × inverse doc freq    | ≤ 5 000    |
| **Bigram TF-IDF** | TF-IDF with unigrams + bigrams   | ≤ 10 000   |

---

## 🤖 Classifiers

1. **Logistic Regression** — linear model with L2 regularisation
2. **Multinomial Naïve Bayes** — probabilistic model, ideal for word counts
3. **Linear SVM** — max-margin linear classifier

---

## 📈 Results

### Test Accuracy Comparison

![Accuracy Comparison](outputs/figures/accuracy_comparison.png)

### Macro F1-Score Comparison

![F1 Comparison](outputs/figures/f1_comparison.png)

### Cross-Validation Accuracy

![CV Accuracy](outputs/figures/cv_accuracy.png)

### Word Clouds

| Sport | Politics |
|-------|----------|
| ![Sport](outputs/figures/wordcloud_sport.png) | ![Politics](outputs/figures/wordcloud_politics.png) |

---

## 📝 Report

The full 5+ page report covering data collection, analysis, methodologies, quantitative comparisons, and limitations is available at:

📄 **[report/report.md](report/report.md)**

---

## ⚙️ Tech Stack

- Python 3.10+
- scikit-learn, pandas, numpy
- matplotlib, seaborn, wordcloud
- NLTK

---

## 📜 License

This project is for academic purposes (NLU course assignment). The BBC News dataset is used under fair academic use.

---

## 🙏 Acknowledgements

- BBC News dataset by Greene & Cunningham (2006), UCD
- Scikit-learn team for the ML library
