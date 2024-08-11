import pandas as pd
import pickle
import numpy as np
import sys

input_file = sys.argv[1]
#input_file = f'employee_data.csv'
output_file = f'output/predicted_data.csv'


df = pd.read_csv(input_file)
# Load the model from the file
with open('./models/lin_reg.bin', 'rb') as file:
    dv, lr = pickle.load(file)

categorical = ['Gender', 'Position']
numerical = ['Experience (Years)']

dicts = df[categorical + numerical].to_dict(orient='records')

X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# Calculate the standard deviation of the predicted durations
std_dev = np.std(y_pred)
print("Standard Deviation of Predicted Durations:", std_dev)

print('predicted mean duration:', y_pred.mean())


df_result = pd.DataFrame()
df_result['ID'] = df['ID']
df_result['predicted_salary'] = y_pred
print (df_result)
