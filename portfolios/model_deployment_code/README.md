# Predict Customer Churn with Clean Code
This is a portfolio.

## Overview
Predicting customer churn with clean code is an important practice for deployment. 
This project is to understand the software engineering best practices such as modular code, logging/testing, refactoring and documenting.

## Files in the Repo
 * [data](./data)
    * [bank_data.csv](./data/bank_data.csv) - customers' bank data information for churn prediction models
 * [images](./images)
     * [eda](./images/eda)
        * [churn_hist.png](./images/eda/churn_hist.png) - histogram of churn (distribution)
        * [customer_age_hist.png](./images/eda/customer_age_hist.png) - histogram of customers' ages (distribution)
        * [heatmap_corr.png](./images/eda/heatmap_corr.png) - heatmap of correlations of features
        * [martial_status_bar.png](./images/eda/martial_status_bar.png) - bar graph of customers' martial status (distribution)
        * [total_trans_ct_density.png](./images/eda/total_trans_ct_density.png)  - density graph of customers' total transactions (distribution)
     * [results](./images/results)
        * [feature_importance_plot.png](./images/results/feature_importance_plot.png) - feature importance plot from Random forest classifier
        * [lrc_rfc_roc_curve.png](./images/results/lrc_rfc_roc_curve.png) - ROC curve plot (Logistic regression vs Random forest classifer)
        * [rfc_shap_bar.png](./images/results/rfc_shap_bar.png) - shap bar graph for Random forest classifier
        * [lr_train_classification_rpt.png](./images/results/lr_train_classification_rpt.png) - classification report from Logistic regression (train set)
        * [lr_test_classification_rpt.png](./images/results/lr_test_classification_rpt.png) - classification report from Logistic regression (test set)
        * [rf_train_classification_rpt.png](./images/results/rf_train_classification_rpt.png) - classification report from Random forest classifier (train set)
        * [rf_test_classification_rpt.png](./images/results/rf_test_classification_rpt.png) - classification report from Random forest classifier (test set)
 * [logs](./logs)
    * [churn_library.log](./logs/churn_library.log) - logging data of SUCCESS/ERROR while running each function in churn_library.py through churn_script_logging_and_tests.py
 * [models](./models)
     * [logistic_model.pkl](./models/logistic_model.pkl) - .pkl of Logistic regression model
     * [rfc_models.pkl](./models/rfc_model.pkl) - .pkl of Random forest classifier model
 * [churn_library.py](./churn_library.py) - script file that runs functions to perform modelling
 * [churn_script_logging_and_tests.py](./churn_script_logging_and_tests.py) - script file that tests/logs SUCCESS/ERROR for each function in churn_library.py
 * [requirements.txt](./requirements.txt) - list of libraries need to be installed to run churn_library.py
 * [README.md](./README.md) - README.md that explains what users need to know before starting

## Dependencies
Make sure to install all necessary dependencies to run main script below.

* pip install -r requirements.txt

## Running main script
This script runs to fulfill following activities:
1. EDA
2. Feature Engineering (including encoding of categorical variables)
3. Model Training
4. Prediction
5. Model Evaluation

* python churn_library.py

## Logging/Testing
This script runs to test following functions:
1. import_data
2. perform_eda
3. encoder_helper
4. perform_feature_engineering
5. train_models

The log file will be in ./logs/churn_library.log and it will log all the SUCCESS/ERROR occurred while running each function.
Note that some functions may be dependent on other functions so that it can save the logs from dependent functions as well and it is intended.
e.g. test_eda(perform_eda) would log from test_import as well since performing EDA would require that data frame was successfully imported.

* python churn_script_logging_and_tests.py