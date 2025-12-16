# 📊 Feature Engineering: Mean Encoding (Target Encoding)

Mean Encoding is a way to convert **high-cardinality categorical features** (like `user_id`, `item_id`) into meaningful numerical values using the **target variable**.

---

## 1. Why Mean Encoding?

### Problem
- One-Hot Encoding breaks when categories are too many
- Creates huge sparse matrices
- Causes overfitting and high memory usage

### Idea
Replace each category with the **average target value** for that category.

**Example:**  
If `item_id = A` usually has high sales, its encoded value will be high.

---

## 2. Main Risk: Data Leakage

If you compute category means using the **same data you encode**, the model indirectly sees the target.

**Result:**  
- Unrealistically high accuracy  
- Poor performance on new data  

👉 Solution: **Never compute the mean using the same rows you encode**

---

## 3. Mean Encoding Schemes (Leak-Safe)

### 1️⃣ K-Fold Mean Encoding (Most Common)

**What it does**
- Split data into `K` folds
- Encode each fold using statistics from the other `K-1` folds

**How to implement**
1. Split data into `K` folds
2. For each fold:
   - Compute category means from other folds
   - Apply them to the current fold

**Why it works**
- Each row is encoded using unseen data

---

### 2️⃣ Leave-One-Out (LOO) Encoding

**What it does**
- For each row, compute the category mean **excluding that row**

**Formula (GitHub-safe LaTeX):**
```latex
$$
\text{LOO}_j = \frac{\sum y_c - y_j}{N_c - 1}
$$
````

Where:

* ( y_j ) = target of current row
* ( N_c ) = count of category ( c )

**Notes**

* Very strong signal
* Expensive for large datasets
* Needs smoothing for rare categories

---

### 3️⃣ Smoothing (Regularized Mean Encoding)

**Why needed**

* Rare categories give noisy means

**Idea**
Blend category mean with global mean.

**Formula:**

```latex
$$
\text{Encoded}(c) =
\frac{\text{Mean}(c) \cdot N_c + \alpha \cdot \text{GlobalMean}}
{N_c + \alpha}
$$
```

**Key points**

* ( \alpha ) controls regularization
* Small count → closer to global mean
* Large count → closer to category mean

---

### 4️⃣ Expanding Mean (Time-Series Only)

**When to use**

* Data has time order (sales, logs, events)

**Rule**

* Use **only past data**
* Never use future information

**How to implement**

1. Sort data by time
2. For each row:

   * Compute mean from earlier rows only

**Why it works**

* Fully prevents look-ahead bias

---

## 4. Practical Details

### Missing Values

* Some categories won’t have a mean
* Fill with:

  * Global mean (recommended)
  * Or a constant

### Evaluation

* Check correlation between:

  * Encoded feature
  * Target variable
* Higher correlation = stronger signal

---

## ✅ Quick Summary

| Method         | When to Use                 | Safe |
| -------------- | --------------------------- | ---- |
| K-Fold         | General datasets            | ✅    |
| LOO            | Large data, many categories | ✅    |
| Smoothing      | Rare categories             | ✅    |
| Expanding Mean | Time-series                 | ✅    |

**Key takeaway:**
Mean Encoding is powerful, but only when **leakage is strictly controlled**.

