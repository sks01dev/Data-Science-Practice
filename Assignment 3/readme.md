# 📊 Feature Engineering: Mean EncodingsThis document outlines the core concepts and implementation steps for **Mean Encoding** (also known as **Target Encoding**), a powerful technique for transforming high-cardinality categorical features into numerical values based on the target variable.

---

##1. The Problem: Encoding High-Cardinality Categoricals###The ChallengeStandard encoding methods like One-Hot Encoding fail when a categorical feature (like `item_id` or `user_id`) has **thousands of unique values** (high cardinality). One-Hot Encoding creates a massive, sparse feature matrix that overfits and consumes too much memory.

###The Solution: Mean EncodingMean Encoding replaces a categorical feature's value with the **average value of the target variable** (y) for that category.

* **Intuition:** The Mean Encoded value numerically captures the **predictive power** of a category toward the target. For example, if the average sales price (`target`) for `item_id A` is consistently high, its Mean Encoding will be high, providing a strong signal to the model.

###The Danger: Data Leakage (Overfitting)The primary risk of Mean Encoding is **data leakage**. If you calculate the target mean using the entire training set and then use that mean to encode the same data, the encoder has already seen the answer it is supposed to predict. This leads to artificially inflated accuracy and severe overfitting.

---

##2. Mean Encoding Schemes (Leak Prevention)To prevent leakage, the mean must be calculated on a subset of the data *different* from the data being encoded. Four common, robust schemes are implemented to manage this:

###Scheme 1: KFold Scheme* **Intuition:** Split the data into K folds (e.g., K=5). The encoding mean for a fold is calculated using the data **outside** that fold (the "out-of-fold" mean).
* **Implementation Steps:**
1. Split the data (e.g., K=5 folds).
2. Iterate through each fold (k):
* Calculate the mean target value for each category using data **outside** fold k.
* Apply that mean to encode the categories **inside** fold k.




* **Result:** The encoding for any data point is based on data it was **not** trained on, preventing leakage.

###Scheme 2: Leave-One-Out Scheme (LOO)* **Intuition:** For every single data point, the encoding mean is calculated based on **all other data points** in the dataset, **excluding the current observation**.
* **Implementation Details:** This is computationally expensive but highly effective. It requires excluding the current observation's target value (y_j) from its own mean calculation.

###Scheme 3: Smoothing Scheme (Regularization)* **Intuition:** This technique stabilizes the encoding for categories with very few observations (low count). It blends the unreliable **category mean** with the stable **global mean**.
* **Formula (Smoothing):**

* **Key Parameters:**
* \text{Mean}(c) and \text{Count}(c): Mean and count for the specific category c.
* \text{Global Mean}: Mean target value across the entire dataset.
* \alpha: **Smoothing factor**. A higher \alpha puts more weight on the global mean, especially when \text{Count}(c) is small.



###Scheme 4: Expanding Mean Scheme (Time-Series Data)* **Intuition:** This is **essential for time-dependent data**. The mean for a given data point can only be calculated using data that occurred **before** it in the timeline (historical data only), completely eliminating look-ahead leakage.
* **Implementation Steps:**
1. **Sort the data** chronologically.
2. For each observation, calculate the target mean for its category using **only the historical data** up to the previous timestamp.


* **Result:** Creates a feature that evolves correctly over time.

---

##3. General Implementation Details* **Fill NaNs:** Missing values (NaNs) in the resulting encoding columns (which occur for categories not present in the mean-calculation set) are typically filled with the **Global Mean** or a pre-defined constant (e.g., 0.3343).
* **Final Output:** The predictive power of each encoding scheme is evaluated by calculating the **correlation coefficient** (\rho) between the resulting numerical encoding column and the target variable.
