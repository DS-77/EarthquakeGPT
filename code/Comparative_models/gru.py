"""
This module implements the Gated Recurrent Unit (GRU) model for forecasting earthquake magnitudes.

Author: Deja S.
Created: 12-11-2024
Edited: 20-11-2024
Version: 1.0.5
"""

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neuralforecast.models import GRU
from sklearn.model_selection import KFold
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, r2_score


def main():
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_data", type=str, required=True, help="The path to the training data")
    ap.add_argument("-v", "--testing_data", type=str, required=True, help="The path to the testing data")

    opts = vars(ap.parse_args())

    # Check if data is available
    training_path = opts["training_data"]
    testing_path = opts["testing_data"]
    weights_path = "run/weights/GRU"
    results_path = "run/results"
    plot_path = "run/five_fold"

    if not os.path.exists(training_path):
        print(f"ERROR: '{training_path}' does not exist.")
        exit()

    if not os.path.exists(testing_path):
        print(f"ERROR: '{testing_path}' does  not exist.")
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
    temp_training_data['ds'] = pd.to_datetime(temp_training_data["Date(YYYY/MM/DD)"] + " " + temp_training_data["Time(UTC)"], errors='coerce')
    temp_training_data.rename(columns={"Magnitude(ergs)": "y"}, inplace=True)
    training_data = temp_training_data[['unique_id', 'ds', 'y', 'Latitude(deg)', 'Longitude(deg)', 'Depth(km)']]

    temp_testing_data['unique_id'] = "earthquake_series"
    temp_testing_data['ds'] = pd.to_datetime(
        temp_testing_data["Date(YYYY/MM/DD)"] + " " + temp_testing_data["Time(UTC)"], errors='coerce')
    temp_testing_data.rename(columns={"Magnitude(ergs)": "y"}, inplace=True)
    testing_data = temp_testing_data[['unique_id', 'ds', 'y', 'Latitude(deg)', 'Longitude(deg)', 'Depth(km)']]

    # Model
    gru_model = None

    # Setting up for Five-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=12)
    avg_MAE = []
    avg_MSE = []
    avg_RMSE = []
    avg_MAPE = []
    avg_r2 = []

    for fold, (train_index, val_index) in enumerate(kf.split(training_data)):
        train_df, val_df = training_data.iloc[train_index], training_data.iloc[val_index]

        gru_model = NeuralForecast(
            models=[GRU(h=577, input_size=6,
                        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
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

        # Train model
        gru_model.fit(train_df)
        pred = pd.DataFrame()

        # Predict values
        for i in range(0, len(val_df), 577):
            temp_val_chunk = val_df.iloc[i:i+577]
            temp_pred = gru_model.predict(temp_val_chunk).reset_index()
            pred = pd.concat([pred, temp_pred], ignore_index=True)

        # Calculate metrics
        if fold == 4:
            # Case for the last k-fold split: 4x Groups of 2885 and 1x Group 2884
            mae = mean_absolute_error(val_df['y'].values, pred['GRU'][:2884].values)
            mse = mean_squared_error(val_df['y'].values, pred['GRU'][:2884].values)
            rmse = root_mean_squared_error(val_df['y'].values, pred['GRU'][:2884].values)
            mape = mean_absolute_percentage_error(val_df['y'].values, pred['GRU'][:2884].values)
            r2 = r2_score(val_df['y'].values, pred['GRU'][:2884].values)
        else:
            mae = mean_absolute_error(val_df['y'].values, pred['GRU'].values)
            mse = mean_squared_error(val_df['y'].values, pred['GRU'].values)
            rmse = root_mean_squared_error(val_df['y'].values, pred['GRU'].values)
            mape = mean_absolute_percentage_error(val_df['y'].values, pred['GRU'].values)
            r2 = r2_score(val_df['y'].values, pred['GRU'].values)

        avg_MSE.append(mse)
        avg_MAE.append(mae)
        avg_RMSE.append(rmse)
        avg_MAPE.append(mape)
        avg_r2.append(r2)

        # Print Results
        print()
        print(f"Fold {fold + 1} Results")
        print("=" * 80)
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAPE: {mape:.3f}")
        print(f"R2: {r2:.3f}")
        print("=" * 80)

    print("-- Saving model weights ...")
    gru_model.save(path=weights_path, model_index=None, overwrite=True, save_dataset=True)

    # Plotting the Five-Fold Results
    # Plot style
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot MSE
    axes[0, 0].plot(range(1, 6), avg_MSE, marker='o', color='b', linestyle='-', label='MSE')
    axes[0, 0].set_title('Mean Squared Error (MSE) for 5 Folds')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].legend()

    # Plot MAE
    axes[0, 1].plot(range(1, 6), avg_MAE, marker='o', color='g', linestyle='-', label='MAE')
    axes[0, 1].set_title('Mean Absolute Error (MAE) for 5 Folds')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()

    # Plot RMSE
    axes[0, 2].plot(range(1, 6), avg_RMSE, marker='o', color='r', linestyle='-', label='RMSE')
    axes[0, 2].set_title('Root Mean Squared Error (RMSE) for 5 Folds')
    axes[0, 2].set_xlabel('Fold')
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].legend()

    # Plot MAPE
    axes[1, 0].plot(range(1, 6), avg_MAPE, marker='o', color='orange', linestyle='-', label='MAPE')
    axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE) for 5 Folds')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].legend()

    # Plot R^2
    axes[1, 1].plot(range(1, 6), avg_r2, marker='o', color='purple', linestyle='-', label='R2')
    axes[1, 1].set_title('R-squared (R2) for 5 Folds')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('R2')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{plot_path}/gru_five_fold_plot.png")
    # plt.show()

    # Print the results
    print()
    print("Training Five Fold Results: GRU")
    print("=" * 80)
    print(f"Average MAE: {np.mean(avg_MAE):.3f}")
    print(f"Average MSE: {np.mean(avg_MSE):.3f}")
    print(f"Average RMSE: {np.mean(avg_RMSE):.3f}")
    print(f"Average MAPE: {np.mean(avg_MAPE):.3f}")
    print(f"Average R2: {np.mean(avg_r2):.3f}")

    # Save results of Five-Fold to file
    ff_filename = f"{results_path}/gru_ff_results.txt"
    with open(ff_filename, 'w') as ff_file:
        ff_file.write("Fold\t|\tMAE\t|\tMSE\t|\tRMSE\t|\tMAPE\t|\tR2\n")
        ff_file.write("-" * 80)
        ff_file.write("\n")

        for i in range(len(avg_MAE)):
            ff_file.write(f"{i+1}\t|\t{avg_MAE[i]:.3f}\t|\t{avg_MSE[i]:.3f}\t|\t{avg_RMSE[i]:.3f}\t|\t{avg_MAPE[i]:.3f}\t|\t{avg_r2[i]:.3f}\n")

        ff_file.write("\n")
        ff_file.write("Training Five Fold Results: GRU\n")
        ff_file.write("=" * 80)
        ff_file.write(f"\nAverage MAE: {np.mean(avg_MAE):.3f}\n")
        ff_file.write(f"Average MSE: {np.mean(avg_MSE):.3f}\n")
        ff_file.write(f"Average RMSE: {np.mean(avg_RMSE):.3f}\n")
        ff_file.write(f"Average MAPE: {np.mean(avg_MAPE):.3f}\n")
        ff_file.write(f"Average R2: {np.mean(avg_r2):.3f}\n")

    # Final Test
    print()
    print("Evaluating the Model on testing data...")

    # Required variables - Dividing the testing samples like the training samples
    test_pred = pd.DataFrame()
    batch_size = 577
    b_num = 7
    start_index = 0

    # Evaluating the model
    for b in range(b_num - 1):
        temp_chunk = testing_data.iloc[start_index:start_index + batch_size]
        temp_pred = gru_model.predict(temp_chunk).reset_index()
        test_pred = pd.concat([test_pred, temp_pred], ignore_index=True)
        start_index += batch_size

    # Evaluating on the remaining testing data
    last_chunk = testing_data.iloc[start_index:]
    last_pred = gru_model.predict(last_chunk)
    last_pred = last_pred.iloc[:144]
    test_pred = pd.concat([test_pred, last_pred], ignore_index=True)

    # Calculate the metrics
    test_mae = mean_absolute_error(testing_data['y'].values, test_pred['GRU'].values)
    test_mse = mean_squared_error(testing_data['y'].values, test_pred['GRU'].values)
    test_rmse = root_mean_squared_error(testing_data['y'].values, test_pred['GRU'].values)
    test_mape = mean_absolute_percentage_error(testing_data['y'].values, test_pred['GRU'].values)
    test_r2 = r2_score(testing_data['y'].values, test_pred['GRU'].values)

    # Save Final result to a file
    final_filename = f"{results_path}/gru_final_result.txt"
    with open(final_filename, 'w') as f_file:
        f_file.write("Testing Results:\n")
        f_file.write("=" * 80)
        f_file.write(f"\nTesting MAE: {test_mae:.3f}\n")
        f_file.write(f"Testing MSE: {test_mse:.3f}\n")
        f_file.write(f"Testing RMSE: {test_rmse:.3f}\n")
        f_file.write(f"Testing MAPE: {test_mape:.3f}\n")
        f_file.write(f"Testing R2: {test_r2:.3f}\n")
        f_file.write("-" * 80)

    print()
    print("Testing Results:")
    print("=" * 80)
    print(f"Testing MAE: {test_mae:.3f}")
    print(f"Testing MSE: {test_mse:.3f}")
    print(f"Testing RMSE: {test_rmse:.3f}")
    print(f"Testing MAPE: {test_mape:.3f}")
    print(f"Testing R2: {test_r2:.3f}")
    print("-" * 80)
    print("Done.")


if __name__ == '__main__':
    main()
