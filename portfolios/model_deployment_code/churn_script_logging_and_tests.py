from pathlib import Path
import os
import logging
#import churn_library_solution as cls
import churn_library as cl
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = test_import(cl.import_data)

    # perform EDA
    perform_eda(df)

    pth = Path("./images/eda")
    file_names = [
        "churn_hist",
        "customer_age_hist",
        "heatmap_corr",
        "martial_status_bar",
        "total_trans_ct_density"]

    # check if file exist in the correct file path
    try:
        for name in file_names:
            file_pth = pth.joinpath(f'{name}.png')
            assert file_pth.is_file()
        logging.info(
            "SUCCESS: All of EDA images are saved in the correct file path!")
    except AssertionError as err:
        logging.error("ERROR: At least one of EDA images is missing!")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # change this flag to test any response
    response = 'Churn'
    df = test_import(cl.import_data)
    df = encoder_helper(df, category_lst, response=response)

    # test when response = 'Churn'
    category_lst_encoded = [
        'Gender_' + response,
        'Education_Level_' + response,
        'Marital_Status_' + response,
        'Income_Category_' + response,
        'Card_Category_' + response
    ]

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: test_encoder_helper dataframe doesn't appear to have rows and columns")
        raise err

    # test if category encoded columns exist after encoding process
    try:
        for c in category_lst_encoded:
            assert c in df.columns
        logging.info("SUCCESS: All encoded categorical columns exist!")
    except AssertionError as err:
        logging.error(
            "ERROR: At least one of encoded categorical columns is missing!")

    return df, category_lst, response


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    encoded = test_encoder_helper(cl.encoder_helper)
    X_train, X_test, y_train, y_test = perform_feature_engineering(*encoded)

    try:
        assert len(y_test) == len(X_test)
        assert len(y_train) == len(X_train)
    except AssertionError as err:
        logging.error(
            "ERROR: At least one of length of train/test set doesn't match!")
        raise err
    logging.info("SUCCESS: All train/test sets match in terms of length!")

    return X_train, X_test, y_train, y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    train_test = test_perform_feature_engineering(
        cl.perform_feature_engineering)

    # train models
    train_models(*train_test)

    pth = Path("./models")
    model_names = ['rfc_model.pkl', 'logistic_model.pkl']

    # check if file exist in the correct file path
    for name in model_names:
        file_pth = pth.joinpath(name)
        try:
            assert file_pth.is_file()
        except AssertionError as err:
            logging.error("ERROR: At least one of models is not saved!")
            raise err
    logging.info("SUCCESS: All models are saved in the correct file path!")


if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
