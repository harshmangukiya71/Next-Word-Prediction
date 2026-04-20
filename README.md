# 🧠 N-Gram Language Model with Advanced Sampling Techniques

A complete implementation of a **probabilistic language model** for next-word prediction using classical NLP methods enhanced with modern sampling strategies like **Top-K, Top-P (Nucleus), and Temperature scaling**.

---

## 🚀 Features

* ✅ Trigram (N-Gram) Language Model
* ✅ Laplace Smoothing
* ✅ Greedy (Argmax) Prediction
* ✅ Top-K Sampling
* ✅ Top-P (Nucleus) Sampling
* ✅ Temperature Scaling
* ✅ CDF-based Probabilistic Sampling

---

## 📊 Problem Statement

Given a sequence of words:

```
(w₁, w₂, ..., wₙ)
```

Predict the next word:

```
wₙ₊₁
```

---

## 🧮 Methodology

### 🔹 N-Gram Model (Trigram)

We model probability as:

```
P(wₙ | wₙ₋₂, wₙ₋₁)
```

---

### 🔹 Laplace Smoothing

To avoid zero probabilities:

```
P(w | c) = (Count(c, w) + 1) / (Count(c) + V)
```

Where:

* `c` = context (previous words)
* `V` = vocabulary size

---

## 🎯 Prediction Strategies

### 1. Greedy (Argmax)

```
w* = argmax P(w | c)
```

* Deterministic
* Often repetitive

---

### 2. Top-K Sampling

* Select top `K` words
* Normalize probabilities
* Sample using CDF

---

### 3. Top-P (Nucleus Sampling)

Select smallest set of words such that:

```
Σ P(w) ≥ p
```

* Adaptive selection
* Better than fixed K

---

### 4. Temperature Scaling

Controls randomness:

```
P_T(w) = P(w)^(1/T) / Σ P(w)^(1/T)
```

| T  | Effect             |
| -- | ------------------ |
| <1 | More deterministic |
| =1 | Normal             |
| >1 | More random        |

---

## 🔥 Final Sampling Pipeline

```
Raw Probabilities
      ↓
Temperature Scaling
      ↓
Sorting
      ↓
Top-P Filtering
      ↓
Normalization
      ↓
CDF Sampling
      ↓
Next Word
```

---

## 🧪 Example

```
Context: ["the", "king"]
```

Possible output:

```
the king was a great ruler of the land
```

---

## 📁 Dataset

* WikiText-2 (HuggingFace)

---

## ▶️ How to Run

```bash
pip install datasets numpy
python your_script.py
```

---

## 📈 Future Improvements

* Kneser-Ney Smoothing
* Backoff Models
* Beam Search
* Repetition Penalty

---

## 🧠 Key Insight

> Instead of always choosing the most probable word, sampling methods allow generating more natural and diverse text.

---


