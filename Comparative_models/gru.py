"""
This module implements the Gated Recurrent Unit (GRU) model for forecasting earthquake magnitudes.

Author: Deja S.
Created: 12-11-2024
Edited: 14-11-2024
Version: 1.0.3
"""

import os
import argparse
import pandas as pd
from neuralforecast.models import GRU
from neuralforecast.auto import AutoGRU
from sklearn.model_selection import KFold
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset
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

    if not os.path.exists(training_path):
        print(f"ERROR: '{training_path}' does not exist.")
        exit()

    if not os.path.exists(testing_path):
        print(f"ERROR: '{testing_path}' does  not exist.")
        exit()

    # Retrieve the data
    temp_training_data = pd.read_csv(training_path, header=0)
    temp_testing_data = pd.read_csv(testing_path, header=0)

    # Parse data
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

    kf = KFold(n_splits=5, shuffle=True, random_state=12)
    avg_MAE = []
    avg_MSE = []
    avg_RMSE = []
    avg_MAPE = []
    avg_r2 = []

    for fold, (train_index, val_index) in enumerate(kf.split(training_data)):
        train_df, val_df = training_data.iloc[train_index], training_data.iloc[val_index]

        # Model Configurations
        # TODO: Break up training data into smaller batches
        gru_model = NeuralForecast(
            models=[GRU(h=12, input_size=6,
                        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                        scaler_type='robust',
                        encoder_n_layers=2,
                        encoder_hidden_size=128,
                        context_size=10,
                        decoder_hidden_size=128,
                        decoder_layers=2,
                        max_steps=5,
                        )
                    ],
            freq='D'
        )

        # Train model
        gru_model.fit(train_df)

        # Predict values
        pred = gru_model.predict(val_df[:12])

        # print(pred['GRU'])
        # print(val_df['y'])
        # exit()

        # Calculate metrics
        mae = mean_absolute_error(val_df['y'][:12].values, pred['GRU'].values)
        mse = mean_squared_error(val_df['y'][:12].values, pred['GRU'].values)
        rmse = root_mean_squared_error(val_df['y'][:12].values, pred['GRU'].values)
        mape = mean_absolute_percentage_error(val_df['y'][:12].values, pred['GRU'].values)
        r2 = r2_score(val_df['y'][:12].values, pred['GRU'].values)

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

        exit()

    # TODO: save trained model
    # TODO: create plot of final result

    # Final Test


if __name__ == '__main__':
    main()
