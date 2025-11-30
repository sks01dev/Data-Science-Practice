
# 🏆 Data Leakage via Graph Structure 

## 🔍 1. Problem Setup

You receive pairs:

```
(FirstId, SecondId) → predict 0/1
```

No features.
But the **pair structure leaks class information**.

---

## 🧠 2. Core Insight

Pairs naturally define an **undirected graph**:

* Nodes = unique IDs
* Edges = observed pairs
* Same-class nodes → similar neighborhoods
* Different-class nodes → different neighborhoods

Graph topology leaks the labels.

---

## 🏗 3. Adjacency Matrix (Sparse)

For **N** unique IDs, build:

```
A = N × N sparse adjacency matrix
A[i, j] = 1 if (i, j) is an observed pair
A is symmetric
```

Construction idea:

```python
row = [FirstId, SecondId]
col = [SecondId, FirstId]
A = coo_matrix((1, (row, col)), shape=(N, N)).tocsr()
A.data[:] = 1
```

---

## 🧬 4. Node Representation

Each row of `A` is a sparse vector:

```
A[i]  →  1 × N vector of neighbors of node i
```

This acts as the **node embedding**.

---

## 🔥 5. Magic Feature: Shared Neighbor Count

For any pair **(i, j)**:

```
f(i, j) = A[i] ⋅ A[j]
```

Expanded:

```
f(i, j) = Σ_k  A[i, k] * A[j, k]
```

Interpretation:

* High `f(i, j)` → many shared neighbors → likely **label = 1**
* Low `f(i, j)` → few shared neighbors → likely **label = 0**

---

## 🎯 6. Threshold Selection (Critical Step)

We must map high/low scores to labels.

Compute:

```
p = positive_rate = accuracy_if_predict_all_ones
```

Then:

1. Sort scores `f` descending
2. Mark top `p × M` pairs as **1**
3. The rest as **0**

Threshold index:

```
k = floor(p * M)
```

Threshold value:

```
T = (f_sorted[k] + f_sorted[k+1]) / 2
```

Prediction rule:

```
predict 1 if f(i, j) ≥ T
else 0
```

---

## 📐 7. Shapes Summary

| Object          | Meaning                   | Shape         |
| --------------- | ------------------------- | ------------- |
| `test`          | input pairs               | (M, 3)        |
| `A`             | adjacency matrix          | (N, N) sparse |
| `rows_FirstId`  | nodes for first elements  | (M, N) sparse |
| `rows_SecondId` | nodes for second elements | (M, N) sparse |
| `f`             | shared neighbor scores    | (M,)          |
| `pred`          | predicted labels          | (M,)          |

---

## 🎓 8. Formula Summary 

Adjacency:

```
A[i, j] = 1  if pair (i, j) exists
```

Shared-neighbor feature:

```
f(i, j) = Σ_k A[i, k] * A[j, k]
```

Threshold:

```
p = positive_rate
k = floor(p * M)
T = (f_sorted[k] + f_sorted[k+1]) / 2
```

---

## 🚀 9. When This Works

Use this method when:

* You have pair data
* IDs appear many times
* Test set is not random
* No meaningful features
* Graph structure creates label leakage

Examples:

* Quora duplicate question detection
* Product / listing matching
* User/session linking
* Image pair verification
* Link prediction with leakage
---
