# Classifying Fraud Through Anomaly Detection

""" Summarizing the BAF dataset revealed an extreme imbalance in the number of fraudulent bank accounts, with approximately 1% 
    (~10,000 records) being fraudulent. Such a small proportion warrants exploring whether an anomaly detection algorithm may 
    adequately classify fraudulent records.
"""

# Imports & Dependencies

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc

# Functions

def create_splits(data):
    """ This function creates train/validate/test splits from the given data. test_size default value is 0.3, val_size default 
        value is 0.1 which equates to a 60/10/30 train/validate/test split. """
    
    # Remove categorical columns
    data.drop(["payment_type", "employment_status", "housing_status", "source", "device_os"],axis=1, inplace = True)
    
    X = data.iloc[:, 1:] # Features
    y = data.iloc[:, 0] # Fraud_bool
    
    # 60% train, 40% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= 0.4, random_state=150, stratify=y)
    
    # 25% of 40% = 10% (validation), 75% of 40% = 30% (test)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size= 0.75, random_state=150, stratify=y_temp)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_model(X):
    """ This function trains an Isolation_Forest classifier from sklearn to detect anomalies. """
    
    # Contamination is the proportion of outliers in the dataset, 1%
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=150) 
    iso_forest.fit(X)
    
    return iso_forest

def calculate_TPR(y_true, y_pred):
    """ This function finds the True Positive Rate at approximately 5% False Positive Rate. """
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fpr_target = 0.05
    index = np.argmin(np.abs(fpr - fpr_target))
    tpr_at_fpr = tpr[index]
    threshold_at_fpr = thresholds[index]
    
    return tpr_at_fpr

def eval_model(model, X, y):
    """ This function evaluates the Isolation_Forest classifier using fraud dection metrics(ie. TPR at an FPR threshold of 5%, and 
        Balanced Accuracy). """
    
    # Predictions
    y_pred = model.predict(X) 
    
    # Relabeling predictions from (-1, 1) anomaly/ normal to (1, 0) for fraud/ non-fraud
    y_pred_binary = np.where(y_pred == -1, 1, 0)

    # Predicted probabilties
    anomaly_scores = -model.decision_function(X)

    # Calculate metrics
    target_tpr = calculate_TPR(y, anomaly_scores)
    auc_score = roc_auc_score(y, anomaly_scores)
    ba_score = balanced_accuracy_score(y, y_pred_binary)
    f_score = f1_score(y, y_pred_binary, pos_label=1)

    return target_tpr, auc_score, ba_score, f_score
    
# Execution

def main():
    # Note: The baf dataset cannot be uploaded to github. It is available through Kaggle.
    baf_df = pd.read_csv("Base.csv")
    
    # Create data splits
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_splits(baf_df)
    print("Splits Completed")
    
    # Train Isolation_Forest
    clf = train_model(X_train)
    print("Training Completed")
    
    # Validate Classifier (training data)
    train_TPR, train_AUC, train_BA, train_F1 = eval_model(clf, X_train, y_train)

    # Validate Classifier (validation data)
    valid_TPR, valid_AUC, valid_BA, valid_F1 = eval_model(clf, X_valid, y_valid)
    print("Validation Completed")

    # Evaluate Classifier (testing data)
    test_TPR, test_AUC, test_BA, test_F1 = eval_model(clf, X_test, y_test)
    print("Testing Completed")

    # Save results
    results_df = pd.DataFrame([
    ["Train", train_TPR, train_AUC, train_BA, train_F1],
    ["Validate", valid_TPR, valid_AUC, valid_BA, valid_F1],
    ["Test", test_TPR, test_AUC, test_BA, test_F1]]
    ,columns=["Data", "True_Positive_Rate", "AUC_Score", "Balanced_Accuracy", "F1_Score"])

    results_df = results_df.round(3)

    #results_df.to_csv("anomaly_results.csv", index=False) # optional
    
if __name__ == "__main__":
    main()