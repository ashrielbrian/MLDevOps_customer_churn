"""
    Module to predict customer churn using Logistic Regresstion
    and Random Forest classifier models.

    Author: Brian Tang
    Date: July 2021
"""
import logging
import logging.config
import yaml
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import IMAGES_DIR, CAT_COLUMNS, KEEP_COLUMNS
sns.set()


with open('log_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    logging.config.dictConfig(config)

logger = logging.getLogger('mainLogger')


def import_data(pth, **kwargs):
    '''
        Returns dataframe for the csv found at pth
        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
    '''
    try:
        logger.info(f'Loading data from {pth}...')
        df = pd.read_csv(pth, **kwargs)
        return df
    except FileNotFoundError as err:
        logger.exception('churn_library')
        logger.error(f'No file found at {pth}. Exiting...')
        raise err


def generate_histograms(df, category_lst, xlabels, target_dir):
    '''
        Saves histograms of dataframe categories
        input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            xlabels: list of strings to set as the histogram xlabel
            target_dir: directory to save histograms to
        output:
            None
    '''
    for cat, xlabel in zip(category_lst, xlabels):
        filename = f'{cat.lower()}_histogram.jpg'
        try:
            logger.info(f'Attempting to generate histogram for {cat}.')
            plt.figure(figsize=(20, 10))
            df[cat].hist()
            plt.ylabel('Number of Customers')
            plt.xlabel(xlabel)
            plt.savefig(f'{target_dir}/{filename}')
            logger.info(f'Saved histogram: {filename}.')
        except KeyError:
            logger.error(f'{cat} category not found in DataFrame.')


def perform_eda(df, histogram_category_lst, target_dir):
    '''
        Perform eda on df and save figures to images folder
        input:
            df: pandas dataframe
            histogram_category_lst: list of columns that contain categorical features
            target_dir: directory to save results to
        output:
            None
    '''
    logger.info('Performing exploratory data analysis...')
    generate_histograms(
        df,
        histogram_category_lst,
        histogram_category_lst,
        target_dir)

    bar_chart_col = 'Marital_Status'
    try:
        ms_filename = f'{bar_chart_col.lower()}_bar_chart.jpg'
        plt.figure(figsize=(20, 10))
        df[bar_chart_col].value_counts('normalize').plot(kind='bar')
        plt.savefig(f'{target_dir}/{ms_filename}')
        logger.info(f'Saved {ms_filename}')
    except KeyError:
        logger.error(f'Column `{bar_chart_col}` not found in DataFrame')

    dist_col = 'Total_Trans_Ct'
    try:
        tt_filename = f'{dist_col.lower()}.jpg'
        plt.figure(figsize=(20, 10))
        sns.distplot(df[dist_col])
        plt.savefig(f'{target_dir}/{tt_filename}')
        logger.info(f'Saved {tt_filename}')
    except KeyError:
        logger.error(f'Column `{dist_col}` not found in DataFrame')

    hm_filename = 'heat_map.jpg'
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{target_dir}/{hm_filename}', bbox_inches='tight')
    logger.info(f'Saved {hm_filename}')


def encoder_helper(df, category_lst, response='Churn'):
    '''
        Helper function to turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the notebook
        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: [optional] string of response used for naming target y column
        output:
                df: pandas dataframe with new columns for
    '''

    df[response] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for cat in category_lst:
        try:
            logger.info(f'Encoding categorical variable: {cat}')
            cat_groups = df.groupby(cat).mean()[response]
            df[cat +
               f'_{response}'] = df[cat].apply(lambda val: cat_groups.loc[val])
        except KeyError:
            logger.error(f'{cat} category not found in DataFrame.')
    return df


def perform_feature_engineering(df, target_feature, test_size=0.3):
    '''
        Runs feature engineering to keep columns and returns train/test data
        input:
            df: pandas dataframe
            response: [optional] string of response used for naming target y column
        output:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    logger.info(
        f'Performing feature engineering. Keeping columns: {", ".join(KEEP_COLUMNS)}.')
    logger.info(f'Target feature is: `{target_feature}`')
    target_df = pd.DataFrame()
    target_label = df[target_feature]
    target_df[KEEP_COLUMNS] = df[KEEP_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        target_df, target_label, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                target_dir):
    '''
        Produces classification report for training and testing results and stores report as image
        in images folder
        input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            target_dir: directory to save image results to
        output:
            None
    '''
    # plots Random Forest model classification report
    rfc_filename = 'random_forest_cr.jpg'
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'{target_dir}/{rfc_filename}', bbox_inches='tight')
    logger.info(f'Saved Random Forest classification report: {rfc_filename}')

    # plots Logistic Regression model classification Report
    lr_filename = 'logistic_regression_cr.jpg'
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'{target_dir}/{lr_filename}', bbox_inches='tight')
    logger.info(
        f'Saved Logistic Regression classification report: {lr_filename}')


def roc_plot(lr_model, rfc_model, x_test, y_test, target_dir):
    """
        Saves the ROC results for the LR and RFC models on the same plot
        input:
            lr_model: training response values
            rfc_model:  test response values
            x_test: training predictions from logistic regression
            y_test: training predictions from random forest
            target_dir: directory to save image results t
        output:
            None
    """
    roc_filename = 'roc_plot.jpg'
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(rfc_model, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig(f'{target_dir}/{roc_filename}')
    logger.info(f'Saved ROC plot: {roc_filename}')


def feature_importance_plot(model, x_data, target_dir):
    '''
        Creates and stores the feature importances in pth
        input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
        output:
            None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    filename = 'random_forest_feature_importance.jpg'
    plt.savefig(f'{target_dir}/{filename}', bbox_inches='tight')
    logger.info(f'Saved feature importance plot: {filename}')


def train_random_forest_classifier(x_train, y_train, param_grid=None):
    """
        Initializes and trains a random forest classifier
        input:
            x_train: X training data
            y_train: y training data
            param_grid: [optional] optional params for the RFC model
        output:
            trained model: sklearn.linear_model.LogisticRegression
    """
    logger.info('Initialising Random Forest classifier...')
    rfc = RandomForestClassifier(random_state=42)
    if param_grid is None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    logger.info('Training RFC. Please wait...')
    cv_rfc.fit(x_train, y_train)
    return cv_rfc


def train_logistic_regression(x_train, y_train, solver='liblinear'):
    """
        initializes and trains a logistic regression model
        input:
            x_train: X training data
            y_train: y training data
            solver: [optional] the solver to be used in the LR model
        output:
            trained model: sklearn.linear_model.LogisticRegression
    """
    logger.info(f'Initialising Logistic Regression with {solver}.')
    lrc = LogisticRegression(solver=solver)
    logger.info('Training LR. Please wait...')
    lrc.fit(x_train, y_train)
    return lrc


def train_models(x_train, y_train):
    '''
        Train, store model results: images + scores, and store models
        input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
        output:
            None
    '''
    cv_rfc = train_random_forest_classifier(x_train, y_train)
    lrc = train_logistic_regression(x_train, y_train)

    # store models
    try:
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        logger.info('Models saved successfully.')
    except Exception as err:
        logger.error('Error saving models. Exiting.')
        logger.exception('churn_library')
        raise err


def load_model(model_pth):
    '''
        Loads the model given its path
        input:
            model_pth: string path to the model's location in the filesystem
        output:
            model: an sklearn model
    '''
    return joblib.load(model_pth)


if __name__ == '__main__':
    # define directory paths
    EDA_DIR = IMAGES_DIR + '/eda'
    RESULTS_DIR = IMAGES_DIR + '/results'

    # preprocess data
    raw_df = import_data('data/bank_data.csv')
    encoded_df = encoder_helper(raw_df, CAT_COLUMNS)
    perform_eda(encoded_df, ['Churn', 'Customer_Age'], EDA_DIR)

    train_data, test_data, train_label, test_label = perform_feature_engineering(
        encoded_df, target_feature='Churn')

    # model training
    train_models(train_data, train_label)

    # reload models for prediction and reporting
    random_forest_model = load_model('./models/rfc_model.pkl')
    regression_model = load_model('./models/logistic_model.pkl')

    # run predictions
    train_label_preds_lr = regression_model.predict(train_data)
    test_label_preds_lr = regression_model.predict(test_data)

    train_label_preds_rf = random_forest_model.predict(train_data)
    test_label_preds_rf = random_forest_model.predict(test_data)

    # reporting and plotting results
    classification_report_image(train_label, test_label, train_label_preds_lr,
                                train_label_preds_rf, test_label_preds_lr,
                                test_label_preds_rf, target_dir=RESULTS_DIR)

    feature_importance_plot(random_forest_model, test_data, target_dir=RESULTS_DIR)
    roc_plot(regression_model, random_forest_model, test_data, test_label, target_dir=RESULTS_DIR)
