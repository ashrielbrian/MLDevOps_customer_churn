import os
import logging
import pandas as pd
import churn_library as lb
import logging
import logging.config
import yaml
from constants import CAT_COLUMNS

with open('log_config.yaml', 'r') as config_file:
	config = yaml.safe_load(config_file)
	logging.config.dictConfig(config)

logger = logging.getLogger('testLogger')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
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
		logger.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, fixture):
	'''
	test perform eda function
	'''

	df = fixture['encoded_dataframe']
	perform_eda(df, ['Churn', 'Customer_Age'], 'images/eda')

	try:
		assert os.path.isfile('images/eda/churn_histogram.jpg')
		assert os.path.isfile('images/eda/customer_age_histogram.jpg')
		assert os.path.isfile('images/eda/marital_status_bar_chart.jpg')
		assert os.path.isfile('images/eda/total_trans_ct.jpg')
		assert os.path.isfile('images/eda/heat_map.jpg')
	except AssertionError as e:
		logger.error('Testing perform_eda: Expected output images not found in target dir.')
		raise e

	logger.info("Testing perform_eda: SUCCESS")


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	y_target = 'Churn'

	df = pd.read_csv('data/bank_data.csv')
	df = encoder_helper(df, CAT_COLUMNS, response=y_target)

	try:
		# check target column exists and only has 1s and 0s
		assert y_target in df.columns
		assert (df[y_target].isin([0,1])).all()
	except AssertionError as e:
		logger.error('Testing encoder_helper: Could not find target column in DataFrame.')
		raise e

	for cat in CAT_COLUMNS:
		try:
			assert f'{cat}_{y_target}' in df.columns
		except AssertionError as e:
			logger.error('Testing encoder_helper: Expected output images not found in target dir.')
			raise e
	logger.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering, fixture):
	'''
	test perform_feature_engineering
	'''
	X_train, X_test, y_train, y_test = perform_feature_engineering(fixture['encoded_dataframe'], fixture['y_target'])
	
	try:
		assert len(X_train) > 0
		assert len(X_test) > 0
		assert len(y_train) > 0
		assert len(y_test) > 0
	except AssertionError as e:
		logger.error("Testing perform_feature_engineering: The train/test data isn't of expected length.")
		raise e
	logger.info('Testing perform_feature_engineering: SUCCESS')

def test_train_models(train_models, fixture):
	'''
	test train_models
	'''
	X_train, X_test, y_train, y_test = fixture['data']
	train_models(X_train, X_test, y_train, y_test)

	try:
		assert os.path.isfile('./models/rfc_model.pkl')
		assert os.path.isfile('./models/logistic_model.pkl')
	except AssertionError as e:
		logger.error('Testing train_models: Expected output models not found in target dir.')
		raise e
	logger.info('Testing train_models: SUCCESS')

def setup_test():
	'''
	sets up test fixture
	'''
	logger.info('Setting up the test...')
	
	y_target = 'Churn'
	df = pd.read_csv('data/bank_data.csv')
	df = lb.encoder_helper(df, CAT_COLUMNS, response=y_target)
	X_train, X_test, y_train, y_test = lb.perform_feature_engineering(df, y_target)

	lr_model = lb.load_model('./models/logistic_model.pkl')
	rfc_model = lb.load_model('./models/rfc_model.pkl')

	# run predictions
	y_train_preds_lr = lr_model.predict(X_train)
	y_test_preds_lr = lr_model.predict(X_test)
	y_train_preds_rf = rfc_model.predict(X_train)
	y_test_preds_rf = rfc_model.predict(X_test)

	try:
		assert len(y_train_preds_lr) > 0
		assert len(y_test_preds_lr) > 0
		assert len(y_train_preds_rf) > 0
		assert len(y_test_preds_rf) > 0
	except AssertionError as e:
		logger.error('Testing setup_test: Predicted y labels are not of expected length.')
		raise e

	logger.info('Test fixture is ready.')
	return {
		'y_target': y_target,
		'encoded_dataframe': df,
		'data': (X_train, X_test, y_train, y_test),
		'models': (lr_model, rfc_model),
		'preds': (y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf)
	}

def test_classification_report(classification_report_image, fixture):
	'''
	test classification_report_image
	'''
	y_train, y_test = fixture['data'][2], fixture['data'][3]
	y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf = fixture['preds']

	classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf, target_dir='./images/results')

	try:
		assert os.path.isfile('./images/results/random_forest_cr.jpg')
		assert os.path.isfile('./images/results/logistic_regression_cr.jpg')
	except AssertionError as e:
		logger.error('Testing classification_report_image: Expected output images not found in target dir.')
		raise e

	logger.info('Testing classification_report_image: SUCCESS')

def test_feature_importance_plot(feature_importance_plot, fixture):
	""" 
	test feature_importance_plot
	"""
	rfc_model = fixture['models'][1]
	X_data = fixture['data'][0]
	feature_importance_plot(rfc_model, X_data, './images/results')

	try:
		assert os.path.isfile('./images/results/random_forest_feature_importance.jpg')
	except AssertionError as e:
		logger.error('Testing feature_importance_plot: Expected output images not found in target dir.')
		raise e
	logger.info('Testing feature_importance_plot: SUCCESS')

def test_roc_plot(roc_plot, fixture):
	""" 
	test roc_plot
	"""
	lr_model, rfc_model = fixture['models'][0], fixture['models'][1]
	X_test, y_test = fixture['data'][1], fixture['data'][3]
	roc_plot(lr_model, rfc_model,X_test, y_test, './images/results')

	try:
		assert os.path.isfile('./images/results/roc_plot.jpg')
	except AssertionError as e:
		logger.error('Testing roc_plot: Expected output images not found in target dir.')
		raise e
	logger.info('Testing roc_plot: SUCCESS') 

if __name__ == "__main__":
	fixture = setup_test()

	# test_import(lb.import_data)
	# test_encoder_helper(lb.encoder_helper)
	# test_eda(lb.perform_eda, fixture)
	# test_perform_feature_engineering(lb.perform_feature_engineering, fixture)
	# test_train_models(lb.train_models, fixture)
	# test_classification_report(lb.classification_report_image, fixture)
	# test_feature_importance_plot(lb.feature_importance_plot, fixture)
	test_roc_plot(lb.roc_plot, fixture)







