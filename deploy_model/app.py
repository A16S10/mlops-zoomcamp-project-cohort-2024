import gradio as gr
import pandas as pd
from sagemaker.sklearn.model import SKLearnPredictor
from sagemaker import Session
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging


# Assuming you already have a trained scaler
scaler = StandardScaler()

# Set up the SageMaker session and predictor
sagemaker_session = Session()
endpoint_name = "Custom-sklearn-model2024-09-15-14-18-31" 
predictor = SKLearnPredictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)

# The predict_salary function that processes the CSV file and predicts salaries
def predict_salary(csvfile):
    data = pd.read_csv(csvfile)

    # Rename the column
    data.rename(columns={'Experience (Years)': 'Experience'}, inplace=True)

    data["Gender"] = data["Gender"].astype('string')
    data["Position"] = data["Position"].astype('string')

    # Apply one-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Removing the target column
    X = data_encoded.drop(columns="Salary")
    scaler.fit(X) 

    # Transforming dataset into the desired format
    new_data = X
    
    new_data_scaled = scaler.transform(new_data)

    # Make predictions using the SageMaker predictor
    predictions = predictor.predict(new_data_scaled)

    # Reversing transformations
    original_data = scaler.inverse_transform(new_data_scaled)
    original_ID = original_data[:, :1].astype(int)

    # Convert array to DataFrame with predicted salaries
    df = pd.DataFrame({
        'ID': original_ID.flatten(),  # Flatten to match the shape
        'Predicted_Salary': predictions
    })

    return df




iface = gr.Interface(
    fn=predict_salary,
    inputs=gr.File(label="Upload Employee Data CSV"),
    outputs=[
        gr.Dataframe(label="Predicted Salaries")
    ],
    title="Employee Salary Prediction",
    description="Upload employee data to predict salaries using a pre-trained linear regression model."
)
# Launch the interface
try:
    iface.launch(server_name="0.0.0.0", server_port=8081)
except Exception as e:
    logger.error(f"An error occurred: {e}")




