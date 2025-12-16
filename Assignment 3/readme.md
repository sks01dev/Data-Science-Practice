#📊 Feature Engineering: Mean EncodingsThis document outlines the core concepts and implementation steps for **Mean Encoding** (also known as Target Encoding), a powerful technique for transforming high-cardinality categorical features into numerical values based on the target variable.

---

##1. The Problem: Encoding High-Cardinality Categoricals###The ChallengeStandard encoding methods like One-Hot Encoding fail when a categorical feature (like `item_id` or `user_id`) has **thousands of unique values** (high cardinality). One-Hot Encoding creates a massive, sparse feature matrix that overfits and consumes too much memory.

###The Solution: Mean EncodingMean Encoding replaces a categorical feature's value with the **average value of the target variable** (y) for that category.

* **Intuition:** If the average purchase price (`target`) for `item_id A` is high, and the average price for `item_id B` is low, the numerical difference in their Mean Encoded values directly captures their predictive power toward the target.

###The Danger: Data Leakage (Overfitting)The primary risk of Mean Encoding is **data leakage**. If you calculate the target mean using the entire training set and then use that mean to encode the same data, the encoder has seen the target variable it is supposed to predict. This leads to severe overfitting.

---

##2. Mean Encoding Schemes (Leak Prevention)To prevent leakage, the mean must be calculated on a subset of the data *different* from the data being encoded. Four common, robust schemes are implemented to manage this:

###Scheme 1: KFold Scheme* **Intuition:** Split the data into K folds. For each fold, calculate the encoding mean based on the **data in the other K-1 folds** (the "out-of-fold" mean).
* **Implementation Steps:**
1. Split the data (e.g., K=5 folds).
2. Iterate through each fold (k):
* Calculate the mean target value for each category using data **outside** fold k.
* Apply that mean to encode the categories **inside** fold k.




* **Result:** The encoding for any data point is based on data it was **not** trained on, preventing leakage.

###Scheme 2: Leave-One-Out Scheme (LOO)* **Intuition:** For every single data point, calculate the encoding mean based on **all other data points** in the dataset.
* **Implementation Details:** This is computationally expensive but highly effective. It requires excluding the current observation's target value (y_i) from its own mean calculation.

###Scheme 3: Smoothing Scheme (Regularization)* **Intuition:** This technique addresses categories with very few observations (low count) where the mean might be unreliable. It blends the category mean with the **global mean** to stabilize the encoding.
* **Formula (Smoothing):**

* **Key Parameters:**
* \text{Mean}(c) and \text{Count}(c): Mean and count for the specific category c.
* \text{Global Mean}: Mean target value across the entire dataset.
* \alpha: **Smoothing factor**. A higher \alpha puts more weight on the stable global mean, especially when \text{Count}(c) is small.



###Scheme 4: Expanding Mean Scheme (Time-Series Data)* **Intuition:** This is essential for **time-dependent data**. The mean for a given data point can only be calculated using data that occurred **before** it in the timeline.
* **Implementation Steps:**
1. **Sort the data** chronologically.
2. For each observation, calculate the target mean for its category using **only the historical data** up to the previous timestamp.


* **Result:** This completely eliminates look-ahead leakage and creates a feature that evolves correctly over time.

---

##3. General Implementation Details* **Fill NaNs:** Missing values (NaNs) in the resulting encoding columns (which often occur for categories not present in the mean-calculation set) are typically filled with the **Global Mean** or a pre-defined constant (e.g., 0.3343).
* **Final Output:** The assignment required calculating the **correlation coefficient** between the resulting numerical encoding column and the target variable to evaluate the scheme's predictive power.
