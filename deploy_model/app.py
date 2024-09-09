import gradio as gr
import pandas as pd
import pickle
import numpy as np
from flask import Flask, jsonify
from threading import Thread
import distutils.util 


import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

logger.info("This is a log message")

# Load the model
with open('./models/lin_reg.bin', 'rb') as file:
    dv, lr = pickle.load(file)

def predict_salary(input_file):
    df = pd.read_csv(input_file)
    
    categorical = ['Gender', 'Position']
    numerical = ['Experience (Years)']
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    std_dev = np.std(y_pred)
    mean_salary = y_pred.mean()

    df_result = pd.DataFrame()
    df_result['ID'] = df['ID']
    df_result['predicted_salary'] = y_pred

    return df_result, std_dev, mean_salary



# Gradio Interface
iface = gr.Interface(
    fn=predict_salary,
    inputs=gr.File(label="Upload Employee Data CSV"),
    outputs=[
        gr.Dataframe(label="Predicted Salaries"),
        gr.Textbox(label="Standard Deviation of Predicted Salaries"),
        gr.Textbox(label="Mean Predicted Salary")
    ],
    title="Employee Salary Prediction",
    description="Upload employee data to predict salaries using a pre-trained linear regression model."
)

# Create Flask app for health check
flask_app = Flask(__name__)

@flask_app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

def start_flask_app():
    print("Starting Flask app for health check on port 5001")
    flask_app.run(host='0.0.0.0', port=8080)



# Start Flask app in a separate thread
flask_thread = Thread(target=start_flask_app)
flask_thread.start()


try:
    iface.launch(server_name="0.0.0.0", server_port=8081)
except Exception as e:
    logger.error(f"An error occurred: {e}")




