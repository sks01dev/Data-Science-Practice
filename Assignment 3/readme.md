# 📊 Feature Engineering: Mean Encodings

This document outlines the core concepts and implementation steps for **Mean Encoding** (also known as **Target Encoding**), a powerful technique for transforming high-cardinality categorical features into numerical values based on the target variable.

---

## 1. The Problem: Encoding High-Cardinality Categoricals

### The Challenge

Standard encoding methods like **One-Hot Encoding** fail when a categorical feature (such as `item_id` or `user_id`) has **thousands of unique values** (high cardinality).

- One-Hot Encoding creates a **massive, sparse feature matrix**
- Leads to **overfitting**
- Consumes excessive **memory and computation**

### The Solution: Mean Encoding

**Mean Encoding** replaces each category with the **average value of the target variable** (`y`) for that category.

**Intuition:**  
The mean-encoded value numerically captures the **predictive power** of a category toward the target.

**Example:**  
If the average sales price (`target`) for `item_id = A` is consistently high, its mean encoding will be high, providing a strong signal to the model.

### The Danger: Data Leakage (Overfitting)

The primary risk of Mean Encoding is **data leakage**.

If you calculate the category mean using the **entire training dataset** and then encode the same data using that mean, the encoder has already seen the answer it is supposed to predict.

**Consequences:**
- Artificially inflated model performance
- Severe overfitting
- Poor generalization to unseen data

---

## 2. Mean Encoding Schemes (Leak Prevention)

To prevent leakage, the mean must be calculated on a subset of the data **different** from the data being encoded.

Below are four robust and commonly used schemes.

---

### Scheme 1: K-Fold Mean Encoding

**Intuition:**  
Split the data into `K` folds (e.g., `K = 5`).  
For each fold, compute category means using **only the data outside that fold** (out-of-fold data).

#### Implementation Steps

1. Split the dataset into `K` folds
2. For each fold `k`:
   - Compute mean target values for each category using data **outside fold k**
   - Apply these means to encode categories **inside fold k**

**Result:**  
Each data point is encoded using information from data it was **not trained on**, preventing leakage.

---

### Scheme 2: Leave-One-Out (LOO) Encoding

**Intuition:**  
For each observation, compute the mean target value for its category using **all other observations**, excluding the current one.

#### Key Idea

For observation \( j \) in category \( c \):

\[
\text{LOO Mean}_j = \frac{\sum_{i \in c} y_i - y_j}{\text{Count}(c) - 1}
\]

**Notes:**
- Highly effective
- Computationally expensive
- Particularly sensitive for categories with very low counts

---

### Scheme 3: Smoothing (Regularized Mean Encoding)

**Intuition:**  
Stabilizes encodings for categories with **few observations** by blending:
- The **category mean**
- The **global mean**

#### Smoothing Formula

\[
\text{Encoded}(c) = 
\frac{\text{Mean}(c) \times \text{Count}(c) + \alpha \times \text{Global Mean}}
{\text{Count}(c) + \alpha}
\]

#### Key Parameters

- `Mean(c)` – Mean target value for category `c`
- `Count(c)` – Number of observations for category `c`
- `Global Mean` – Mean target value across the entire dataset
- `α (alpha)` – **Smoothing factor**
  - Larger `α` → more weight on global mean
  - Especially useful when `Count(c)` is small

---

### Scheme 4: Expanding Mean Encoding (Time-Series Data)

**Intuition:**  
For time-dependent data, the encoding must use **only past information** to avoid look-ahead bias.

#### Implementation Steps

1. **Sort the dataset chronologically**
2. For each observation:
   - Compute the category mean using **only historical data**
   - Exclude the current and future observations

**Result:**  
Creates a feature that evolves correctly over time and fully avoids temporal leakage.

---

## 3. General Implementation Details

### Handling Missing Values

- Encoded values may be `NaN` for:
  - Categories not present in the training subset
- Common strategies:
  - Fill with **Global Mean**
  - Fill with a constant (e.g., `0.3343`)

### Model Evaluation

- The effectiveness of each encoding scheme is often evaluated using:
  - **Correlation coefficient** (\( \rho \)) between:
    - Encoded feature
    - Target variable

A higher correlation typically indicates stronger predictive power.

---

## ✅ Summary

| Scheme | Best Use Case | Leakage Safe |
|------|--------------|--------------|
| K-Fold | General tabular data | ✅ |
| Leave-One-Out | High-cardinality, large datasets | ✅ |
| Smoothing | Rare categories | ✅ |
| Expanding Mean | Time-series data | ✅ |

Mean Encoding is powerful—but only when **leakage is carefully controlled**.
