"""
This .py is a library of functions to find customers who are likely to churn.
It saves model relevent files in designated folders.

Author: Sang Yoon Hwang
Date: May 25, 2023
"""

# import libraries
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report
import matplotlib.pyplot as plt
import dataframe_image as dfi
import pandas as pd
import shap
import joblib
import numpy as np
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # create directory called images if it does not exist
    pth = './images/eda'
    is_path = os.path.exists(pth)
    if not is_path:
        os.makedirs(pth)

    # create a column called Churn for EDA
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # churn histogram
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(pth + '/churn_hist.png')

    # customer age histogram
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(pth + '/customer_age_hist.png')

    # martial status bar plot
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(pth + '/martial_status_bar.png')

    # total trans ct density plot
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(pth + '/total_trans_ct_density.png')

    # heatmap correlation plot
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(pth + '/heatmap_corr.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # create a column called Churn for Encoder
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # if response is None, then Churn will be the name of response variable
    if response is None:
        resp_col = 'Churn'
    else:
        resp_col = response

    for cat in category_lst:
        cat_groups = df.groupby(cat).mean()[resp_col]
        cat_lst = [cat_groups.loc[val] for val in df[cat]]
        cat_col = cat + '_' + resp_col
        df[cat_col] = cat_lst

    return df


def perform_feature_engineering(df, category_lst, response):
    '''
    input:
              df: pandas dataframe
              category_lst: list of columns that contain categorical features
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # set X and y
    y = df[response]
    X = encoder_helper(df, category_lst, response)

    # only keep columns that are needed for modelling
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X = X[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # create directory called images if it does not exist
    pth = './images/report'
    is_path = os.path.exists(pth)
    if not is_path:
        os.makedirs(pth)

    # save as images
    report = classification_report(y_test, y_test_preds_rf, output_dict=True)
    df = pd.DataFrame(report).transpose()
    dfi.export(df, pth + '/rf_test_classification_rpt.png')

    report2 = classification_report(
        y_train, y_train_preds_rf, output_dict=True)
    df2 = pd.DataFrame(report2).transpose()
    dfi.export(df2, pth + '/rf_train_classification_rpt.png')

    report3 = classification_report(y_test, y_test_preds_lr, output_dict=True)
    df3 = pd.DataFrame(report3).transpose()
    dfi.export(df3, pth + '/lr_test_classification_rpt.png')

    report4 = classification_report(
        y_train, y_train_preds_lr, output_dict=True)
    df4 = pd.DataFrame(report4).transpose()
    dfi.export(df4, pth + '/lr_train_classification_rpt.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # create directory called models if it does not exist
    is_path = os.path.exists(output_pth)
    if not is_path:
        os.makedirs(output_pth)

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save as image
    plt.savefig(
        output_pth +
        '/feature_importance_plot.png',
        bbox_inches='tight')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # create directory called models if it does not exist
    pth = './models'
    is_path = os.path.exists(pth)
    if not is_path:
        os.makedirs(pth)

    # fit models with best parameters from GridSearchCV
    rfc = RandomForestClassifier(
        criterion='entropy',
        max_depth=100,
        n_estimators=200,
        random_state=42)
    lrc = LogisticRegression(max_iter=1000, solver='saga')
    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # save model scores as images
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(pth + '/lrc_rfc_roc_curve.png')
    plt.clf()

    explainer = shap.TreeExplainer(rfc)
    shap_values = explainer.shap_values(X_test)
    fig = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(pth + '/rfc_shap_bar.png', bbox_inches='tight')
    plt.clf()

    # save the best models
    joblib.dump(rfc, pth + '/rfc_model.pkl')
    joblib.dump(lrc, pth + '/logistic_model.pkl')


if __name__ == "__main__":
    # category lists
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    # Import data
    df = import_data(r"./data/bank_data.csv")
    # Perform EDA
    perform_eda(df)
    # Categorical variable encoding - helper function
    df_cleaned = encoder_helper(df, category_lst, response=None)
    # Train/Test split after feature engineering (select only necessary
    # columns for modelling)
    df_cleaned_final = perform_feature_engineering(
        df_cleaned, category_lst, response='Churn')
    # Train models
    train_models(*df_cleaned_final)
