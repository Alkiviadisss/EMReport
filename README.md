# EMReport

A lightweight Python utility for generating clean, readable evaluation reports for **Regression**, **Classification**, and **Clustering** machine learning models. All in one place.

![EMReport_demo.gif](https://github.com/user-attachments/assets/dbba2ab5-7346-4055-9571-d8cb14c0b75d)

---

## Features

- **Regression Report:** MAE, MSE, RMSE, and R²
- **Classification Report:**  Accuracy, Precision, Recall, F1 Score, and Confusion Matrix
- **Clustering Report:** Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index

---

## Requirements

- Python 3.7+
- `numpy`
- `scikit-learn`

Install dependencies with:

```bash
pip install EMReport
```

---

## Usage

Import the functions you need directly from `EMReport.py`:

```python
from EMReport import EMReport_Regression, EMReport_Classification, EMReport_Clustering
```

---

###  Regression

```python
from EMReport import EMReport_Regression

y_true = [3.0, 5.0, 2.5, 7.0]
y_pred = [2.8, 4.9, 2.7, 6.8]

EMReport_Regression(y_true, y_pred)
```

**Output:**
```
--------------------------------------------------
            REGRESSION EVALUATION REPORT
--------------------------------------------------
Mean Absolute Error (MAE):       0.1500
Mean Squared Error (MSE):        0.0250
Root Mean Squared Error (RMSE):  0.1581
R-squared (R2):                  0.9921
--------------------------------------------------
```

---

###  Classification

```python
from EMReport import EMReport_Classification

y_true = [0, 1, 2, 1, 0]
y_pred = [0, 1, 1, 1, 0]

EMReport_Classification(y_true, y_pred)
```

**Output:**
```
--------------------------------------------------
            CLASSIFICATION EVALUATION REPORT
--------------------------------------------------
Accuracy:                        0.8000
Precision:                       0.8000
Recall:                          0.8000
F1 Score:                        0.8000
--------------------------------------------------
Confusion Matrix:
 [[2 0 0]
  [0 2 0]
  [0 1 0]]
--------------------------------------------------
```

---

###  Clustering

```python
from EMReport import EMReport_Clustering
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
labels = [0, 0, 0, 1, 1, 1]

EMReport_Clustering(X, labels)
```

**Output:**
```
--------------------------------------------------
          CLUSTERING EVALUATION REPORT
--------------------------------------------------
Number of Clusters:        2
Silhouette Score:          0.8571 (Closer to 1 is better)
Davies-Bouldin Index:      0.2041 (Lower is better)
Calinski-Harabasz Index:   90.0000 (Higher is better)
--------------------------------------------------
```

>**Note:** The clustering report requires at least **2 distinct clusters** and fewer clusters than total samples. An error message will be shown if this condition is not met.

---

## API Reference

### `EMReport_Regression(y_true, y_pred)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like | Ground truth target values |
| `y_pred` | array-like | Estimated target values from the model |

---

### `EMReport_Classification(y_true, y_pred)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array-like | Ground truth class labels |
| `y_pred` | array-like | Predicted class labels from the model |

> Uses `weighted` averaging for Precision, Recall, and F1 Score to handle class imbalance.

---

### `EMReport_Clustering(X, labels)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like of shape `(n_samples, n_features)` | Input feature matrix used for clustering |
| `labels` | array-like | Cluster labels assigned to each sample |

---

## Metrics Explained

| Metric | Task | Interpretation |
|--------|------|----------------|
| MAE | Regression | Average absolute difference between predictions and actuals |
| MSE | Regression | Average squared difference, penalises large errors more |
| RMSE | Regression | Square root of MSE, same unit as target variable |
| R² | Regression | Proportion of variance explained (1.0 = perfect fit) |
| Accuracy | Classification | Fraction of correctly classified samples |
| Precision | Classification | Of all positive predictions, how many were correct |
| Recall | Classification | Of all actual positives, how many were found |
| F1 Score | Classification | Harmonic mean of Precision and Recall |
| Silhouette Score | Clustering | Measures how similar a point is to its own cluster vs. others (-1 to 1) |
| Davies-Bouldin Index | Clustering | Average similarity between clusters, lower = better separation |
| Calinski-Harabasz Index | Clustering | Ratio of between-cluster to within-cluster dispersion, higher = better |

---

## License

This project is open-source and available under the [MIT License](LICENSE).
