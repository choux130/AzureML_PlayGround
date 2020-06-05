import sys as s
print(sys.path)

import os
import argparse

import azureml.core
print("SDK version:", azureml.core.VERSION)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from azureml.core import Dataset
from sklearn.externals import joblib

import numpy as np

""" 00 -- Create an empty directory and handling input arguments
"""
print('[Start] 00 -- Create an empty directory and handling input arguments')
os.makedirs('./outputs', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--alpha_start', type=int)
parser.add_argument('--alpha_end', type=int)
parser.add_argument('--alpha_by', type=float)
parser.add_argument('--data-folder', type=str)
args = parser.parse_args()

run = Run.get_context()
print('Data folder is at:', args.data_folder)
print('List all files: ', os.listdir(args.data_folder))

""" 01 -- Load Data
"""
print('[Start] 01 -- Load Data')
X = np.load(os.path.join(args.data_folder, 'features.npy'))
y = np.load(os.path.join(args.data_folder, 'labels.npy'))
print(X.shape)
print(y.shape)

""" 02 -- Data Preparation
"""
print('[Start] 02 -- Data Preparation')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

""" 03 -- Train the model
"""
print('[Start] 04 -- Train the model')
alphas = np.arange(args.alpha_start, args.alpha_end, args.alpha_by)

for alpha in alphas:
    # Use Ridge algorithm to create a regression model
    reg = Ridge(alpha=alpha)
    reg.fit(data["train"]["X"], data["train"]["y"])

    preds = reg.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    run.log('alpha', alpha)
    run.log('mse', mse)

    model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
    with open(model_file_name, "wb") as file:
        joblib.dump(value=reg, filename='outputs/' + model_file_name)

    print('alpha is {0:.2f}, and mse is {1:0.2f}'.format(alpha, mse))