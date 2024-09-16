import argparse
import os
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the environment variables
logger.info(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR')}")
logger.info(f"SM_CHANNEL_TRAIN: {os.environ.get('SM_CHANNEL_TRAIN')}")
logger.info(f"SM_CHANNEL_TEST: {os.environ.get('SM_CHANNEL_TEST')}")

# Check if files exist
train_file = os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'train_v1.csv')
test_file = os.path.join(os.environ['SM_CHANNEL_TEST'], 'test_v1.csv')

if not os.path.exists(train_file):
    logger.error(f"Training file not found at {train_file}")
    exit(1)

if not os.path.exists(test_file):
    logger.error(f"Test file not found at {test_file}")
    exit(1)

logger.info("Files found, proceeding with training.")

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == '__main__':
    

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()




    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--train_file', type=str, default='train_v1.csv')
    parser.add_argument('--test_file', type=str, default='test_v1.csv')

    args, _ = parser.parse_known_args()

    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

  
    features = list(train_df.columns[:-1])
    label = train_df.columns[-1]

    print("Building training and testing datasets")
    scaler=StandardScaler()
    
    X_train = train_df[features]
    X_train=scaler.fit_transform(X_train)

    X_test = test_df[features]
    X_test=scaler.transform(X_test)
    
    y_train = train_df[label]
    y_test = test_df[label]

    print("Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)


    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at " + model_path)
    print()

    y_pred_test = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Test Mean Absolute Error: {test_mae}")