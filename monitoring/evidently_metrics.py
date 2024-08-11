import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
import pickle

from prefect import task, flow
from sklearn.feature_extraction import DictVectorizer

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, RegressionQualityMetric, ColumnQuantileMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	month varchar(3),
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	median_fare_amount float,
	rmse float
)
"""

reference_data = pd.read_csv('data/reference.csv')
with open('models/lin_reg.bin', 'rb') as f_in:
	model = joblib.load(f_in)
	
#with open('../deploy_model/models/lin_reg.bin', 'rb') as file:
#    dv, lr = pickle.load(file)
	
raw_data = pd.read_csv('employee_data_with_month.csv')

#Need to read the data day by day, from March 1, 2024
cat_features = ['Gender', 'Position']
num_features = ['Experience (Years)']

target = "Salary"
column_mapping = ColumnMapping(
    target=target,
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    RegressionQualityMetric(),
    ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)
])

#@task
def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)

#@task
def calculate_metrics_postgresql(curr, month):
    #filter datetime
    current_data = raw_data[raw_data['Month'] == month]
	
    print(current_data)
    cur_dicts = current_data[cat_features + num_features].fillna(0).to_dict(orient='records')
    #X_cur = dv.transform(cur_dicts)
	
    current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))
    #y_pred = lr.predict(X_cur)
    #print(y_pred)
    
    report.run(reference_data = reference_data, current_data = current_data,
        column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    median_experience = result['metrics'][4]['result']['current']['value']
    rmse = result['metrics'][3]['result']['current']['rmse']

    curr.execute(
        "insert into dummy_metrics(month, prediction_drift, num_drifted_columns, share_missing_values, median_fare_amount, rmse) values (%s, %s, %s, %s, %s, %s)",
        (month, prediction_drift, num_drifted_columns, share_missing_values, median_experience, rmse)
    )

#@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		months = ['Jan', 'Feb', 'Mar', 'Apr']
        #We work with 4 months only for now
		for month in months:
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, month)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()
