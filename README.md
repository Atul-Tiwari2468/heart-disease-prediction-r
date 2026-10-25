# Heart Disease Prediction Using R

## Project Overview
This project applies predictive analytics and machine learning to the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. Four classification models are compared to evaluate their ability to distinguish between records with and without recorded heart disease.

## Objectives
- Audit and preprocess the dataset.
- Perform exploratory data analysis (EDA).
- Train and evaluate multiple classification models.
- Compare model performance using classification metrics.

## Tools and Technologies
- R
- caret and recipes
- tidyverse and ggplot2
- randomForest
- rpart
- e1071 and kernlab
- pROC

## Methodology
1. Dataset loading and data quality checks
2. Missing-value analysis
3. Exploratory data analysis and visualization
4. Stratified 80:20 train-test split
5. Preprocessing using imputation, dummy encoding, zero-variance filtering, and normalization
6. Repeated 5-fold cross-validation with two repeats
7. Test-set evaluation using Accuracy, Precision, Recall, Specificity, F1-score, and ROC-AUC

## Test Results

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 81.36% | 0.9120 |
| SVM (Radial) | 81.36% | 0.8958 |
| Random Forest | 81.36% | 0.8947 |
| Decision Tree | 76.27% | 0.7778 |

Logistic Regression achieved the highest ROC-AUC, while SVM achieved the highest recall at the selected 0.50 probability threshold.

## Dataset
UCI Machine Learning Repository — Heart Disease  
https://archive.ics.uci.edu/dataset/45/heart+disease

## How to Run
1. Install R and RStudio.
2. Open `predictive_analytics_heart_disease_corrected.R`.
3. Run the script from the first line.
4. Review the generated results in the `project_outputs` directory.

## Disclaimer
This is an educational machine learning project using a small public dataset. It is not a clinically validated diagnostic system.
