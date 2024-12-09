"""
This module implements the Long Short-Term Memory Recurrent Neural Network (LSTM) model for forecasting earthquake
magnitudes.

Author: Deja S.
Created: 06-11-2024
Edited: 06-11-2024
Version: 1.0.0
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neuralforecast.models import LSTM
from neuralforecast import NeuralForecast
from sklearn.model_selection import TimeSeriesSplit
from neuralforecast.losses.pytorch import DistributionLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, r2_score


def train_lstm_model(train_df, val_df):
    """
    Train the Vanilla Transformer model and predict values.
    :param train_df: (pd.DataFrame) Training data
    :param val_df: (pd.DataFrame) Validation data
    :returns: tuple: Predictions and the trained model
    """
    h = 577
    lstm_model = NeuralForecast(
        models=[LSTM(h=577, input_size=6,
                     loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                     scaler_type='robust',
                     encoder_n_layers=4,
                     encoder_hidden_size=128,
                     context_size=10,
                     decoder_hidden_size=128,
                     decoder_layers=4,
                     max_steps=1000,
                     batch_size=32,
                     )
                ],
        freq='D'
    )

    # Train model
    lstm_model.fit(train_df, val_size=577)
    pred = pd.DataFrame()

    # Predict values
    for i in range(0, len(val_df), 577):
        temp_val_chunk = val_df.iloc[i:i + 577]
        temp_pred = lstm_model.predict(temp_val_chunk).reset_index()
        pred = pd.concat([pred, temp_pred], ignore_index=True)

    return pred, lstm_model


def five_fold_cross_validation(training_data, weights_path, plot_path, results_path):
    """
    Perform five-fold time series cross-validation on the training data.
    This method ensures that the splits respect the temporal order of the data.

    :param training_data: (pd.DataFrame) Input training data
    :param weights_path: (str) Path to save model weights
    :param plot_path: (str) Path to save performance plots
    :param results_path: (str) Path to save results
    :return: tuple: Average performance metrics and the last trained model
    """
    # Setting up Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize lists to store performance metrics
    avg_MAE, avg_MSE, avg_RMSE, avg_MAPE, avg_r2 = [], [], [], [], []
    last_trained_model = None

    # Iterate through the time series cross-validation splits
    for fold, (train_index, val_index) in enumerate(tscv.split(training_data)):
        # Split the data while maintaining temporal order
        train_df = training_data.iloc[train_index]
        val_df = training_data.iloc[val_index]

        # Train and predict using the time mixer model
        pred, lstm_model = train_lstm_model(train_df, val_df)
        last_trained_model = lstm_model

        # Calculate performance metrics
        # Handle potential length discrepancies with val_df and pred
        y_true = val_df['y'].values
        y_pred = pred['LSTM'][:len(y_true)]

        # Calculate and store performance metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Append metrics for averaging
        avg_MSE.append(mse)
        avg_MAE.append(mae)
        avg_RMSE.append(rmse)
        avg_MAPE.append(mape)
        avg_r2.append(r2)

        # Print Results for each fold
        print(f"\nFold {fold + 1} Results")
        print("=" * 80)
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAPE: {mape:.3f}")
        print(f"R2: {r2:.3f}")
        print("=" * 80)

    # Save model weights
    print("-- Saving model weights ...")
    last_trained_model.save(path=weights_path, model_index=None, overwrite=True, save_dataset=True)

    # Plotting the Time Series Cross-Validation Results
    plot_five_fold_performance(avg_MAE, avg_MSE, avg_RMSE, avg_MAPE, avg_r2, plot_path)

    # Save results to file
    save_five_fold_results(avg_MAE, avg_MSE, avg_RMSE, avg_MAPE, avg_r2, results_path)
    return np.mean(avg_MAE), np.mean(avg_MSE), np.mean(avg_RMSE), np.mean(avg_MAPE), np.mean(avg_r2), last_trained_model


def plot_five_fold_performance(avg_MAE, avg_MSE, avg_RMSE, avg_MAPE, avg_r2, plot_path):
    """
    Plot performance metrics for five-fold cross-validation.
    :param avg_MAE: (list) Mean Absolute Error for each fold
    :param avg_MSE: (list) Mean Squared Error for each fold
    :param avg_RMSE: (list) Root Mean Squared Error for each fold
    :param avg_MAPE: (list) Mean Absolute Percentage Error for each fold
    :param avg_r2: (list) R-squared for each fold
    :param plot_path: (str) Path to save the performance plot
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot metrics
    metrics_data = [
        (avg_MSE, 'MSE', 'b', axes[0, 0]),
        (avg_MAE, 'MAE', 'g', axes[0, 1]),
        (avg_RMSE, 'RMSE', 'r', axes[0, 2]),
        (avg_MAPE, 'MAPE', 'orange', axes[1, 0]),
        (avg_r2, 'R2', 'purple', axes[1, 1])
    ]

    for data, name, color, ax in metrics_data:
        ax.plot(range(1, 6), data, marker='o', color=color, linestyle='-', label=name)
        ax.set_title(f'{name} for 5 Folds')
        ax.set_xlabel('Fold')
        ax.set_ylabel(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{plot_path}/lstm_five_fold_plot.png")


def save_five_fold_results(avg_MAE, avg_MSE, avg_RMSE, avg_MAPE, avg_r2, results_path):
    """
    Save five-fold cross-validation results to a file.
    :param avg_MAE: (list) Mean Absolute Error for each fold
    :param avg_MSE: (list) Mean Squared Error for each fold
    :param avg_RMSE: (list) Root Mean Squared Error for each fold
    :param avg_MAPE: (list) Mean Absolute Percentage Error for each fold
    :param avg_r2: (list) R-squared for each fold
    :return: results_path (str) Path to save the results file
    """
    ff_filename = f"{results_path}/lstm_ff_results.txt"
    with open(ff_filename, 'w') as ff_file:
        ff_file.write("Fold\t|\tMAE\t|\tMSE\t|\tRMSE\t|\tMAPE\t|\tR2\n")
        ff_file.write("-" * 80)
        ff_file.write("\n")

        for i in range(len(avg_MAE)):
            ff_file.write(
                f"{i + 1}\t|\t{avg_MAE[i]:.3f}\t|\t{avg_MSE[i]:.3f}\t|\t{avg_RMSE[i]:.3f}\t|\t{avg_MAPE[i]:.3f}\t|\t{avg_r2[i]:.3f}\n")

        ff_file.write("\n")
        ff_file.write("Training Five Fold Results: LSTM\n")
        ff_file.write("=" * 80)
        ff_file.write(f"\nAverage MAE: {np.mean(avg_MAE):.3f}\n")
        ff_file.write(f"Average MSE: {np.mean(avg_MSE):.3f}\n")
        ff_file.write(f"Average RMSE: {np.mean(avg_RMSE):.3f}\n")
        ff_file.write(f"Average MAPE: {np.mean(avg_MAPE):.3f}\n")
        ff_file.write(f"Average R2: {np.mean(avg_r2):.3f}\n")


def evaluate_on_test_data(lstm_model, testing_data, results_path):
    """
    Evaluate the trained model on testing data.
    :param lstm_model: (NeuralForecast) Trained model
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
        temp_pred = lstm_model.predict(temp_chunk).reset_index()
        test_pred = pd.concat([test_pred, temp_pred], ignore_index=True)
        start_index += batch_size

    # Evaluating on the remaining testing data
    last_chunk = testing_data.iloc[start_index:]
    last_pred = lstm_model.predict(last_chunk)
    last_pred = last_pred.iloc[:144]
    test_pred = pd.concat([test_pred, last_pred], ignore_index=True)

    # Calculate the metrics
    test_mae = mean_absolute_error(testing_data['y'].values, test_pred['LSTM'].values)
    test_mse = mean_squared_error(testing_data['y'].values, test_pred['LSTM'].values)
    test_rmse = root_mean_squared_error(testing_data['y'].values, test_pred['LSTM'].values)
    test_mape = mean_absolute_percentage_error(testing_data['y'].values, test_pred['LSTM'].values)
    test_r2 = r2_score(testing_data['y'].values, test_pred['LSTM'].values)

    # Save Final result to a file
    test_results = {
        'mae': test_mae,
        'mse': test_mse,
        'rmse': test_rmse,
        'mape': test_mape,
        'r2': test_r2
    }

    final_filename = f"{results_path}/lstm_final_result.txt"
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
    ap.add_argument("-w", "--weights", type=str, required=False, default="runs/LSTM/weights",
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
    weights_path = "runs/LSTM/weights/"
    results_path = "runs/LSTM/results"
    plot_path = "runs/LSTM/five_fold"

    if not os.path.exists(training_path) and mode == "train":
        print(f"ERROR: '{training_path}' does not exist.")
        exit()

    if not os.path.exists(testing_path) and mode == "test":
        print(f"ERROR: '{testing_path}' does  not exist.")
        exit()

    if not os.path.exists(pretrain_weights) and mode == "test":
        print(f"ERROR: '{pretrain_weights}' does  not exist.")
        exit()

    # Create required directories if not available
    if not os.path.exists("runs"):
        os.mkdir("runs")

    if not os.path.exists(weights_path) and mode == "train":
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
        print("\n--- Starting Five-Fold Cross-Validation ...")
        avg_MAE, avg_MSE, avg_RMSE, avg_MAPE, avg_r2, trained_model = five_fold_cross_validation(
            training_data,
            weights_path,
            plot_path,
            results_path
        )

        # Print Final Cross-Validation Results
        print("\nFinal Five-Fold Cross-Validation Results:")
        print("=" * 50)
        print(f"Average MAE: {np.mean(avg_MAE):.3f}")
        print(f"Average MSE: {np.mean(avg_MSE):.3f}")
        print(f"Average RMSE: {np.mean(avg_RMSE):.3f}")
        print(f"Average MAPE: {np.mean(avg_MAPE):.3f}")
        print(f"Average R2: {np.mean(avg_r2):.3f}")

        print("Done.")

    elif mode == "test":
        lstm_model = NeuralForecast.load(pretrain_weights)
        testing_results = evaluate_on_test_data(lstm_model, testing_data, results_path)
        print("Done.")
    else:
        print(f"ERROR: Can not recognise mode: '{mode}'")
        exit()


if __name__ == '__main__':
    main()
