# 🏆 Data Leakage via Graph Structure

## 🔍 1. Problem Setup

We observe pairs:

```
(FirstId, SecondId) → predict 0/1
```

No features.
But **pairs themselves leak class structure**.

---

## 🧠 2. Core Insight

Pairs implicitly define an **undirected graph**:

* **Nodes** = unique IDs
* **Edges** = observed pairs
* **Same-class items** → dense, similar neighborhoods
* **Different-class items** → sparse, dissimilar neighborhoods

Thus labels leak through **graph topology**.

---

## 🏗 3. Adjacency Matrix (Sparse)

For **N** unique IDs, build a sparse adjacency matrix **A**:

* Shape: **N × N**
* Symmetric
* `A[i, j] = 1` if `(i, j)` is an observed pair

**Construction:**

```python
row = [FirstId, SecondId]
col = [SecondId, FirstId]

A = coo_matrix((1, (row, col)), shape=(N, N)).tocsr()
A.data[:] = 1
```

Matrix must be **sparse** because N can be large (e.g., 26k × 26k).

---

## 🧬 4. Node Representation

Row **A[i]** is a **binary sparse vector** representing neighbors of node *i*.

* Shape: **(1 × N)**

This functions as a **graph embedding**.

---

## 🔥 5. Magic Feature: Shared Neighbor Count

For any pair **(i, j)**:

[
f(i, j) = A[i] \cdot A[j]
= \sum_k A[i,k],A[j,k]
]

Interpretation:

* **High f** → many shared neighbors → likely class **1**
* **Low f** → few shared neighbors → likely class **0**

Output shape:

* **f: (M,)** where M = number of test pairs

These values typically form **two clusters**, e.g., around 14 vs 20.

---

## 🎯 6. Threshold Selection (Critical)

We must decide **which cluster** corresponds to label = 1.

Compute:

[
p = \text{accuracy(all 1s)} = \text{fraction of positives in the test set}
]

Then:

1. Sort scores **f** descending
2. Assign **1** to the top **p × M** pairs
3. Assign **0** to the rest

Threshold:

[
k = \lfloor pM \rfloor
]

[
T = \frac{f_{\text{sorted}}[k] + f_{\text{sorted}}[k+1]}{2}
]

Pairs with `f ≥ T` → predict **1**, else **0**.

---

## 📐 7. Shapes Summary

| Object          | Meaning                | Shape         |
| --------------- | ---------------------- | ------------- |
| `test`          | input pairs            | (M, 3)        |
| `A`             | adjacency matrix       | (N, N) sparse |
| `rows_FirstId`  | node embeddings        | (M, N) sparse |
| `rows_SecondId` | node embeddings        | (M, N) sparse |
| `f`             | shared-neighbor scores | (M,)          |
| `pred`          | final labels           | (M,)          |

---

## 🎓 8. General Formula Summary

### **Adjacency:**

[
A[i,j] = 1 \quad \text{if pair}(i,j)
]

### **Shared Neighbor Feature:**

[
f(i,j) = \sum_k A[i,k]A[j,k]
]

### **Thresholding:**

[
p = \text{positive rate}, \qquad
k = \lfloor pM \rfloor
]

[
T = \frac{f_{\text{sorted}}[k] + f_{\text{sorted}}[k+1]}{2}
]

---

## 🚀 9. Where This Applies

Use this method when:

* You have **pair-based prediction** tasks
* IDs appear multiple times
* Graph structure is implicit
* Test set is **not** randomly sampled
* No/weak features are available
* Score histogram shows **two clusters**

Typical applications:

* Quora duplicate question detection
* Product matching / entity resolution
* Image pair verification
* Session or user linking
* Link prediction with leakage

---
