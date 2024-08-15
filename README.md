# mlops-zoomcamp-project-cohort-2024
## 📝 Description
This is my DataTalksClub MlOps zoomcamp project repo. 

It uses simple employee dataset from IT industry (presumably from India) and tries to predict salary. 
Prediction is mainly based on position & experience.

Dataset credit: https://www.kaggle.com/datasets/abhayayare/employee-data-simulation-it-industry

Here are more details about all the submission topics

## 1. Basic jupyter notebook 
[salary-prediction.ipynb](./salary-prediction.ipynb) is simple salary prediction jupyter notebook.
It will parse [employee_data.csv](./employee_data.csv) via `pandas` and use Linear Regression model from `scikit-learn`.
It will also plot simple histogram of salary distribution using `matplotlib`.

## 2. Experiment Tracking via mlflow
Switch to folder [experiment_tracking](./experiment_tracking)
It has some pre-run experiments. you can simply observe them.

Or You can run these experiments via foll. python scripts. 
### 1. Simple single 'RandomForestRegressor' experiment 
Run train.py
![single-expt.png](./pictures/single-expt.png)
### 2. Multiple experiments with hyperopt parameter tuning 
Run hpo.py
![random-forest-hyperopt.png](./pictures/random-forest-hyperopt.png)
### 3. Register best model with the lowest test RMSE in model registry  
Run register_model.py
![best-model-expt.png](./pictures/best-model-expt.png)
![best-model-registered.png](./pictures/best-model-registered.png)

## 3. Pipeline orchestration via mage
Pipeline code is in folder [model_training_pipeline](./model_training_pipeline)

Here we use `mage` to set pipeline with foll. stages
### 1. ingest (Data loader)
Read employee data csv file
### 2. train_lr (Transformer)
Train linear regression model to predict salary 
### 3. save_to_mlflow (Data exporter)
Save model as mlflow artifact

![mage.png](./pictures/mage.png)

## 4. Deploy batch model via docker
This is a docker image.

It takes input: location of file which has Gender,Experience (Years),Position (Keep this file in local dir)

It ouputs: Predicted salary for all rows in input file 

Instructions to run:

Switch to folder [deploy_model](./deploy_model)

Pull docker container from public docker image at github registry

```bash
docker pull ghcr.io/nilarte/simple_salary_prediction
```

Note: You will need docker login to ghcr.io

ex:
```bash
docker login ghcr.io -u Your_Username
```

Or simply build local image using Dockerfile

Then run:
```bash
docker run -v $(pwd)/test1.csv:/app/test1.csv ghcr.io/nilarte/simple_salary_prediction test1.csv
```

![deploy-model.png](./pictures/deploy-model.png)

### Pending / To Do:
Write simpler shell using docker-compose file. 
It will handle volume mount to input file.
So user can run simpler command.

Try webservice deployment

## 5. Monitoring
Here we monitor key performance metrics of our employee salary predictionusing `Evidently`, 
including prediction drift, rmse and median employee experince . The results are visualized through `Grafana`.

Switch to folder [monitoring](./monitoring)
Here we have [baseline_model_employee_salary_data.ipynb](./monitoring/baseline_model_employee_salary_data.ipynb) 

It will log the following Evidently metrics inline in the notebook

ColumnDriftMetric(For column "prediction")

DatasetDriftMetric

DatasetMissingValuesMetric

RegressionQualityMetric

ColumnQuantileMetric(Median of employee experience in dataset)

Now we shall observe simillar metrics in Grafana dashboard.

Here we have "mock" split our employee data in to 4 month parts in 2024: 

Jan Feb March and April.

So that we can observe metrics chages over these 4 months in dashboard.

Instruction to observe monitoring dashboard:
Please run
```bash
docker-compose up
```
It will start postgres db to store our metrics.
It will also start Grafana ui at http://localhost:3000/


You can observer pre-made metrics dashboard.
Or run [evidently_metrics.py](./monitoring/evidently_metrics.py)
This script will populate metrics data in postgres db.
We can add postgres db as datasouce and create dashboard from postgres data.

        time         |  prediction_drift  | num_drifted_columns | share_missing_values | median_experience |        rmse
---------------------+--------------------+---------------------+----------------------+-------------------+--------------------
 2024-01-01 00:00:00 | 0.8398708501969149 |                   1 |                    0 |                 8 | 22892.770295572125
 2024-02-01 00:00:00 | 0.8933338427537486 |                   1 |                    0 |              10.5 | 26598.315880818354
 2024-03-01 00:00:00 | 0.2853697742641565 |                   0 |                    0 |                 9 | 23442.389799655462
 2024-04-01 00:00:00 | 0.7783973524742297 |                   0 |                    0 |                11 | 26836.565137309874
(4 rows)









 
