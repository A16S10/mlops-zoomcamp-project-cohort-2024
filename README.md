# mlops-zoomcamp-project-cohort-2024
## 📝 Description
This is my DataTalksClub MlOps zoomcamp project repo. 

It uses simple employee dataset from IT industry (presumably from India) and tries to predict salary. 
Prediction is mainly based on position & experience.

Dataset credit: https://www.kaggle.com/datasets/abhayayare/employee-data-simulation-it-industry

Here are more details about all the submission topics

## 1. Basic jupyter notebook 
[salary-prediction.ipynb](./salary-prediction.ipynb) is simple salary prediction jupyter notebook.
It will parse [employee_data.csv](./employee_data.csv) via 'pandas' and use Linear Regression model from 'scikit-learn'.
It will also plot simple histogram of salary distribution using 'matplotlib'.

## 2. Experiment Tracking via mlflow
Switch to folder [experiment_tracking](./experiment_tracking)
It has some prerun experiments 
You can simply start 
Now you can run and/or observe: 
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

## 3. Pipeline orchestartion via mage
Pipeline code is in folder [model_training_pipeline](./model_training_pipeline)

Here we use 'mage' to set pipeline with foll. stages
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

docker pull ghcr.io/nilarte/simple_salary_prediction

Note: You will need docker login to ghcr.io

ex: docker login ghcr.io -u Your_Username

Or simply build local image using Dockerfile

Then run:

docker run -v $(pwd)/test1.csv:/app/test1.csv ghcr.io/nilarte/simple_salary_prediction test1.csv

![deploy-model.png](./pictures/deploy-model.png)

### Pending / To Do:
Write simpler shell using docker-compose file. 
It will handle volume mount to input file.
So user can run simpler command.

Try webservice deployment

## 5. Monitoring
Here we 







 
