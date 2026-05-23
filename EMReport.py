import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def EMReport_Regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print("-" * 50)
    print("            REGRESSION EVALUATION REPORT")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE):       {mae:.4f}")
    print(f"Mean Squared Error (MSE):        {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
    print(f"R-squared (R2):                  {r2:.4f}")
    print("-" * 50)

def EMReport_Classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print("-" * 50)
    print("            CLASSIFICATION EVALUATION REPORT")
    print("-" * 50)
    print(f"Accuracy:                        {acc:.4f}")
    print(f"Precision:                       {prec:.4f}")
    print(f"Recall:                          {rec:.4f}")
    print(f"F1 Score:                        {f1:.4f}")
    print("-" * 50)
    print("Confusion Matrix:\n", cm)
    print("-" * 50)

def EMReport_Clustering(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) == len(X):
        print("-" * 50)
        print("          CLUSTERING EVALUATION REPORT")
        print("-" * 50)
        print("Error: Metrics require at least 2 distinct clusters and fewer clusters than samples.")
        print(f"Number of clusters found: {len(unique_labels)}")
        print("-" * 50)

        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        print("-" * 50)
        print("          CLUSTERING EVALUATION REPORT")
        print("-" * 50)
        print(f"Number of Clusters:        {len(unique_labels)}")
        print(f"Silhouette Score:          {sil_score:.4f} (Closer to 1 is better)")
        print(f"Davies-Bouldin Index:      {db_score:.4f} (Lower is better)")
        print(f"Calinski-Harabasz Index:   {ch_score:.4f} (Higher is better)")
        print("-" * 50)