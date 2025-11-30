# 🏆 Structural Data Leakage: Master Revision Guide

This guide summarizes the methodology used in the **Data Leakages** assignment. The key objective was to exploit a hidden structural flaw in the test data to achieve near-perfect classification accuracy **without using the training set**.

***

## 1. The Core Leak Intuition: Structural Consistency

The competition required predicting a binary label ($y \in \{0, 1\}$) for pairs of objects.

* **The Flaw:** The test set contained numerous instances where physically distinct pairs of objects were **mathematically identical** in their structural feature profile.
* **The Leak Rule:** If two pairs of objects, **Pair A** and **Pair B**, interact with the rest of the dataset in exactly the same way (they share the same "neighborhood" profile), then they **must share the same true classification label** ($y_A = y_B$).

We solved the problem as a **pattern matching exercise**, not a traditional machine learning task.

***

## 2. Methodology: Graph Representation & Feature Engineering

The solution centers on modeling the pairs as an implicit graph structure to derive a powerful predictive feature: the **Shared Neighbor Count**.

### Step 1: Construct the Graph Structure (Adjacency Matrix)

We represent the entire network of relationships using an Adjacency Matrix.

* **Nodes:** The unique IDs of the objects involved.
* **Edges:** An observed pair of objects (an entry in the data).
* **Tool:** The sparse matrix format (`coo_matrix` or `csr_matrix`) is used due to the massive, sparse nature of the potential connections.

| Structure | Goal | Shape |
| :--- | :--- | :--- |
| **Adjacency Matrix ($A$)** | Encodes all observed pairs: $A[i, j] = 1$ if $(i, j)$ is a pair. | $(N_{\text{unique IDs}}, N_{\text{unique IDs}})$ |
| **Construction (Conceptual):** | Set $A[i, j]=1$ and $A[j, i]=1$ for every pair $(i, j)$ to ensure symmetry. | `scipy.sparse.coo_matrix` |

### Step 2: The Magic Feature (Shared Neighbor Count)

For every pair $(i, j)$ that needs to be scored, we calculate how many objects are connected to *both* $i$ and $j$.

* **Feature Intuition:** A high count of shared neighbors implies a **stronger structural similarity**, which correlates directly with a positive label ($\mathbf{1}$).
* **Formula (The Core Calculation, $f$):**
    The shared neighbor count is calculated by the **dot product** of the two nodes' rows in the Adjacency Matrix:
    $$f(i,j) = \mathbf{A}[i] \cdot \mathbf{A}[j] = \sum_k \mathbf{A}[i,k]\mathbf{A}[j,k]$$
* **Resulting Score Shape ($f$):** $(M_{\text{test pairs}},)$

***

## 3. Leak Exploitation: Thresholding and Prediction

We exploit the known phenomenon that the simple score achieved by predicting "all $\mathbf{1}$s" reveals the true positive rate, which serves as the threshold boundary.

### Step 3: Threshold Selection

1.  **Calculate Positive Rate ($p$):** The fraction of positive labels in the test set is equivalent to the accuracy obtained by simply predicting $\mathbf{1}$ for everything. This value is obtained from the competition's leak/scoring.
    $$p = \text{positive rate} \approx \text{accuracy}(\text{predict all 1s})$$
2.  **Sort Scores:** Sort the shared neighbor scores ($f$) for all test pairs in descending order.
3.  **Find Cutoff Index ($k$):** Determine the index where the fraction $p$ is reached.
    $$k = \lfloor p \cdot M_{\text{test pairs}} \rfloor$$
4.  **Set Threshold ($T$):** The threshold is the boundary value separating the top $k$ scores from the rest.
    $$T = (\mathbf{f}_{\text{sorted}}[k] + \mathbf{f}_{\text{sorted}}[k+1]) / 2$$

### Step 4: Final Prediction

* **Prediction Rule:** Any pair with a shared neighbor count $f(i, j)$ greater than the threshold $T$ is predicted as $\mathbf{1}$, and all others are predicted as $\mathbf{0}$. This single classification rule achieves near-perfect accuracy.

| Object | Meaning | Shape |
| :--- | :--- | :--- |
| $\mathbf{A}$ | Adjacency Matrix | $(N_{\text{unique IDs}}, N_{\text{unique IDs}})$ Sparse |
| $\mathbf{f}$ | Shared Neighbor Count Scores | $(M_{\text{test pairs}},)$ |
| $\mathbf{y}_{\text{pred}}$ | Final Binary Labels | $(M_{\text{test pairs}},)$ |

