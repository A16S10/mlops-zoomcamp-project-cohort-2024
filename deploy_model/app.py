import gradio as gr
import pandas as pd
import pickle
import numpy as np
import distutils.util 

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

iface.launch(server_name="0.0.0.0", server_port=7861)

