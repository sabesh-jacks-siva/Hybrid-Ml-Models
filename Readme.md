# Hybrid Machine Learning Models

This repository contains various hybrid machine learning models implemented in Python. These models combine multiple machine learning algorithms to leverage their strengths and improve performance.

## Hybrid Models

1. **Stacking (Ensemble of Logistic Regression, Decision Tree, and SVM)**
   - Script: `stacking.py`
   - Combines Logistic Regression, Decision Tree, and Support Vector Machine (SVM) using a Stacking Classifier.

2. **Blending (Ensemble of KNN, Random Forest, and Gradient Boosting)**
   - Script: `blending.py`
   - Combines K-Nearest Neighbors (KNN), Random Forest, and Gradient Boosting using a blending technique.

3. **Voting Classifier (Ensemble of Naive Bayes, SVM, and XGBoost)**
   - Script: `voting_classifier.py`
   - Combines Naive Bayes, Support Vector Machine (SVM), and XGBoost using a Voting Classifier.

4. **PCA with Random Forest**
   - Script: `pca_random_forest.py`
   - Applies Principal Component Analysis (PCA) for dimensionality reduction and trains a Random Forest on the reduced data.

5. **Hybrid LSTM and Random Forest for Time Series Forecasting**
   - Script: `lstm_random_forest.py`
   - Combines Long Short-Term Memory (LSTM) neural network and Random Forest for time series forecasting.

## Requirements

- pandas
- scikit-learn
- matplotlib
- tensorflow (for LSTM)
- xgboost (for XGBoost)

Install the requirements using pip:

```bash
pip install pandas scikit-learn matplotlib tensorflow xgboost
