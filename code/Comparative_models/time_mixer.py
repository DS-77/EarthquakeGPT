"""
This module implements the TimeMixer model for forecasting earthquake magnitudes.

Author: Deja S.
Created: 07-12-2024
Edited: 07-12-2024
Version: 1.0.0
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from neuralforecast.models import TimeMixer
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, r2_score


def train_timer_mixer(train_df, weights_path):
    h = 577
    time_mixer_model = NeuralForecast(models=[TimeMixer(h=h, input_size=6,
                    loss=DistributionLoss(),
                    enable_progress_bar = True,
                    scaler_type='robust',
                    encoder_n_layers=2,
                    encoder_hidden_size=128,
                    context_size=10,
                    decoder_hidden_size=128,
                    decoder_layers=2,
                    max_steps=1,
                    batch_size=32,
                    )
                ],
        freq='D'
    )

    # Cross-validation
    preds = time_mixer_model.cross_validation(train_df, step_size=h, n_windows=5, verbose=0, refit=True)

    # Calculate metrics
    mae = mean_absolute_error(train_df['y'].values, preds['TimeMixer'].values)
    mse = mean_squared_error(train_df['y'].values, preds['TimeMixer'].values)
    rmse = root_mean_squared_error(train_df['y'].values, preds['TimeMixer'].values)
    mape = mean_absolute_percentage_error(train_df['y'].values, preds['TimeMixer'].values)
    r2 = r2_score(train_df['y'].values, preds['TimeMixer'].values)

    # Print Results
    print(f"\nFold Results")
    print("=" * 80)
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.3f}")
    print(f"R2: {r2:.3f}")
    print("=" * 80)

    # Save the model
    print("-- Saving model weights ...")
    time_mixer_model.save(path=weights_path, model_index=None, overwrite=True, save_dataset=True)

    return mae, mse, rmse, mape, r2


def evaluate_on_test_data(time_mixer_model, testing_data, results_path):
    """
    Evaluate the trained model on testing data.
    :param time_mixer_model: (NeuralForecast) Trained model
    :param testing_data: (pd.DataFrame) Testing dataset
    :param results_path: (str) Path to save test results
    :return: dict Testing performance metrics
    """
    # Required variables - Dividing the testing samples like the training samples
    test_pred = pd.DataFrame()
    batch_size = 577
    b_num = 7
    start_index = 0

    # Evaluating the model
    for b in range(b_num - 1):
        temp_chunk = testing_data.iloc[start_index:start_index + batch_size]
        temp_pred = time_mixer_model.predict(temp_chunk).reset_index()
        test_pred = pd.concat([test_pred, temp_pred], ignore_index=True)
        start_index += batch_size

    # Evaluating on the remaining testing data
    last_chunk = testing_data.iloc[start_index:]
    last_pred = time_mixer_model.predict(last_chunk)
    last_pred = last_pred.iloc[:144]
    test_pred = pd.concat([test_pred, last_pred], ignore_index=True)

    # Calculate the metrics
    test_mae = mean_absolute_error(testing_data['y'].values, test_pred['TimeMixer'].values)
    test_mse = mean_squared_error(testing_data['y'].values, test_pred['TimeMixer'].values)
    test_rmse = root_mean_squared_error(testing_data['y'].values, test_pred['TimeMixer'].values)
    test_mape = mean_absolute_percentage_error(testing_data['y'].values, test_pred['TimeMixer'].values)
    test_r2 = r2_score(testing_data['y'].values, test_pred['TimeMixer'].values)

    # Save Final result to a file
    test_results = {
        'mae': test_mae,
        'mse': test_mse,
        'rmse': test_rmse,
        'mape': test_mape,
        'r2': test_r2
    }

    final_filename = f"{results_path}/time_mixer_final_result.txt"
    with open(final_filename, 'w') as f_file:
        f_file.write("Testing Results:\n")
        f_file.write("=" * 80)
        for metric, value in test_results.items():
            f_file.write(f"\nTesting {metric.upper()}: {value:.3f}")
        f_file.write("\n" + "-" * 80)

    print("\nTesting Results:")
    print("=" * 80)
    for metric, value in test_results.items():
        print(f"Testing {metric.upper()}: {value:.3f}")
    print("-" * 80)

    return test_results


def main():
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_data", type=str, required=True, help="The path to the training data")
    ap.add_argument("-v", "--testing_data", type=str, required=True, help="The path to the testing data")
    ap.add_argument("-m", "--mode", type=str, required=True, help="The mode: 'train' or 'test'")
    # TODO: Add pre-trained weights path here.
    ap.add_argument("-w", "--weights", type=str, required=False, default="run/weights/transformer",
                    help="The path to pre-trained weights.")
    opts = vars(ap.parse_args())

    # Variables
    os.environ['NIXTLA_ID_AS_COL'] = '1'
    os.environ['PL_DISABLE_FORK'] = '1'
    mode = opts['mode']

    # Check if data is available
    training_path = opts["training_data"]
    testing_path = opts["testing_data"]
    pretrain_weights = opts["weights"]
    weights_path = "run/weights/transformer"
    results_path = "run/results"
    plot_path = "run/five_fold"

    if not os.path.exists(training_path):
        print(f"ERROR: '{training_path}' does not exist.")
        exit()

    if not os.path.exists(testing_path):
        print(f"ERROR: '{testing_path}' does  not exist.")
        exit()

    if not os.path.exists(pretrain_weights):
        print(f"ERROR: '{pretrain_weights}' does  not exist.")
        exit()

    # Create required directories if not available
    if not os.path.exists(weights_path):
        os.mkdir("run")
        os.makedirs(weights_path)
        os.mkdir(results_path)
        os.mkdir(plot_path)

    # Retrieve the data
    temp_training_data = pd.read_csv(training_path, header=0)
    temp_testing_data = pd.read_csv(testing_path, header=0)

    # Parse the data
    temp_training_data['unique_id'] = "earthquake_series"
    temp_training_data['ds'] = pd.to_datetime(
        temp_training_data["Date(YYYY/MM/DD)"] + " " + temp_training_data["Time(UTC)"], errors='coerce')
    temp_training_data.rename(columns={"Magnitude(ergs)": "y"}, inplace=True)
    training_data = temp_training_data[['unique_id', 'ds', 'y', 'Latitude(deg)', 'Longitude(deg)', 'Depth(km)']]

    temp_testing_data['unique_id'] = "earthquake_series"
    temp_testing_data['ds'] = pd.to_datetime(
        temp_testing_data["Date(YYYY/MM/DD)"] + " " + temp_testing_data["Time(UTC)"], errors='coerce')
    temp_testing_data.rename(columns={"Magnitude(ergs)": "y"}, inplace=True)
    testing_data = temp_testing_data[['unique_id', 'ds', 'y', 'Latitude(deg)', 'Longitude(deg)', 'Depth(km)']]

    if mode == "train":
        train_mae, train_mse, train_rmse, train_mape, train_r2 = train_timer_mixer(training_data, weights_path)

    elif mode == "test":
        time_mixer_model = NeuralForecast.load(pretrain_weights)
        test_results = evaluate_on_test_data(time_mixer_model, testing_data, results_path)

    else:
        print(f"ERROR: Can not recognise mode: '{mode}'")
        exit()


if __name__ == "__main__":
    main()