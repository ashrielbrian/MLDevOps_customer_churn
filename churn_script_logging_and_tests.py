""" 
	Module to run unit tests on churn_library.py

	Author: Brian Tang
	Date: July 2021
"""
import os
import logging
import logging.config
import yaml
import pandas as pd
import churn_library as lb
from constants import CAT_COLUMNS

with open('log_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    logging.config.dictConfig(config)

logger = logging.getLogger('testLogger')


def test_import(import_data):
    """
        Tests import_data
        input:
            import_data: function to be tested
        output:
            None
    """
    try:
        df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, fixture):
	"""
        Tests perform_eda
        input:
            perform_eda: function to be tested
			fixture: dictionary containing test components
        output:
            None
    """

    df = fixture['encoded_dataframe']
    perform_eda(df, ['Churn', 'Customer_Age'], 'images/eda')

    try:
        assert os.path.isfile('images/eda/churn_histogram.jpg')
        assert os.path.isfile('images/eda/customer_age_histogram.jpg')
        assert os.path.isfile('images/eda/marital_status_bar_chart.jpg')
        assert os.path.isfile('images/eda/total_trans_ct.jpg')
        assert os.path.isfile('images/eda/heat_map.jpg')
    except AssertionError as err:
        logger.error(
            'Testing perform_eda: Expected output images not found in target dir.')
        raise err

    logger.info("Testing perform_eda: SUCCESS")


def test_encoder_helper(encoder_helper):
    """
        Tests encoder_helper
        input:
            encoder_helper: function to be tested
        output:
            None
    """
    y_target = 'Churn'

    df = pd.read_csv('data/bank_data.csv')
    df = encoder_helper(df, CAT_COLUMNS, response=y_target)

    try:
        # check target column exists and only has 1s and 0s
        assert y_target in df.columns
        assert (df[y_target].isin([0, 1])).all()
    except AssertionError as err:
        logger.error(
            'Testing encoder_helper: Could not find target column in DataFrame.')
        raise err

    for cat in CAT_COLUMNS:
        try:
            assert f'{cat}_{y_target}' in df.columns
        except AssertionError as err:
            logger.error(
                'Testing encoder_helper: Expected output images not found in target dir.')
            raise err
    logger.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering, fixture):
    """
        Tests perform_feature_engineering
        input:
            perform_feature_engineering: function to be tested
			fixture: dictionary containing test components
        output:
            None
    """
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        fixture['encoded_dataframe'], fixture['y_target'])

    try:
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logger.error(
            "Testing perform_feature_engineering: The train/test data isn't of expected length.")
        raise err
    logger.info('Testing perform_feature_engineering: SUCCESS')


def test_train_models(train_models, fixture):
    """
        Tests train_models
        input:
            train_models: function to be tested
			fixture: dictionary containing test components
        output:
            None
    """
    x_train, x_test, y_train, y_test = fixture['data']
    train_models(x_train, y_train)

    try:
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
    except AssertionError as err:
        logger.error(
            'Testing train_models: Expected output models not found in target dir.')
        raise err
    logger.info('Testing train_models: SUCCESS')


def setup_test():
    '''
    	Sets up the test fixtures
		input:
			None
		output:
			fixture: dictionary containing test components
    '''
    logger.info('Setting up the test...')

    y_target = 'Churn'
    df = pd.read_csv('data/bank_data.csv')
    df = lb.encoder_helper(df, CAT_COLUMNS, response=y_target)
    x_train, x_test, y_train, y_test = lb.perform_feature_engineering(
        df, y_target)

    lr_model = lb.load_model('./models/logistic_model.pkl')
    rfc_model = lb.load_model('./models/rfc_model.pkl')

    # run predictions
    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    try:
        assert len(y_train_preds_lr) > 0
        assert len(y_test_preds_lr) > 0
        assert len(y_train_preds_rf) > 0
        assert len(y_test_preds_rf) > 0
    except AssertionError as err:
        logger.error(
            'Testing setup_test: Predicted y labels are not of expected length.')
        raise err

    logger.info('Test fixture is ready.')
    return {
        'y_target': y_target, 'encoded_dataframe': df, 'data': (
            x_train, x_test, y_train, y_test), 'models': (
            lr_model, rfc_model), 'preds': (
                y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf)}


def test_classification_report(classification_report_image, fixture):
    """
        Tests classification_report_image
        input:
            classification_report_image: function to be tested
			fixture: dictionary containing test components
        output:
            None
    """
    y_train, y_test = fixture['data'][2], fixture['data'][3]
    y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf = fixture['preds']

    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf, target_dir='./images/results')

    try:
        assert os.path.isfile('./images/results/random_forest_cr.jpg')
        assert os.path.isfile('./images/results/logistic_regression_cr.jpg')
    except AssertionError as err:
        logger.error(
            'Testing classification_report_image: Expected output images not found in target dir.')
        raise err

    logger.info('Testing classification_report_image: SUCCESS')


def test_feature_importance_plot(feature_importance_plot, fixture):
    """
        Tests feature_importance_plot
        input:
            feature_importance_plot: function to be tested
			fixture: dictionary containing test components
        output:
            None
    """
    rfc_model = fixture['models'][1]
    x_test = fixture['data'][1]
    feature_importance_plot(rfc_model, x_test, './images/results')

    try:
        assert os.path.isfile(
            './images/results/random_forest_feature_importance.jpg')
    except AssertionError as err:
        logger.error(
            'Testing feature_importance_plot: Expected output images not found in target dir.')
        raise err
    logger.info('Testing feature_importance_plot: SUCCESS')


def test_roc_plot(roc_plot, fixture):
    """
        Tests roc_plot
        input:
            roc_plot: function to be tested
			fixture: dictionary containing test components
        output:
            None
    """
    lr_model, rfc_model = fixture['models'][0], fixture['models'][1]
    x_test, y_test = fixture['data'][1], fixture['data'][3]
    roc_plot(lr_model, rfc_model, x_test, y_test, './images/results')

    try:
        assert os.path.isfile('./images/results/roc_plot.jpg')
    except AssertionError as err:
        logger.error(
            'Testing roc_plot: Expected output images not found in target dir.')
        raise err
    logger.info('Testing roc_plot: SUCCESS')


if __name__ == "__main__":
    test_fixture = setup_test()

    test_import(lb.import_data)
    test_encoder_helper(lb.encoder_helper)
    test_eda(lb.perform_eda, test_fixture)
    test_perform_feature_engineering(lb.perform_feature_engineering, test_fixture)
    test_train_models(lb.train_models, test_fixture)
    test_classification_report(lb.classification_report_image, test_fixture)
    test_feature_importance_plot(lb.feature_importance_plot, test_fixture)
    test_roc_plot(lb.roc_plot, test_fixture)
