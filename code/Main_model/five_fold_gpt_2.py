"""
This module is the training (fine-tuning) and inferencing code for GPT 2 model.

Author: Deja S.
Version: 1.0.8
Created: 30-10-2024
Edited: 07-12-2024
"""

import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, r2_score
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2Config, GPT2Model, PreTrainedModel


class EarthquakeGPT(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, gpt_model, dropout_rate=0.1):
        super().__init__(gpt_model.config)
        self.gpt = gpt_model

        # More robust regression head with dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(gpt_model.config.n_embd, gpt_model.config.n_embd // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(gpt_model.config.n_embd // 2, 1)
        )
        self.criterion = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)[0]
        last_hidden_state = gpt_output[:, -1, :]
        dropout_output = self.dropout(last_hidden_state)
        pred_mag = self.regression_head(dropout_output)

        loss = torch.tensor(0.0, device=pred_mag.device)
        if labels is not None:
            loss = self.criterion(pred_mag.squeeze(-1), labels)

        return {
            "loss": loss,
            "logits": pred_mag
        }


class EarthquakeDataset(Dataset):
    def __init__(self, data, tokeniser, max_length=128):
        self.data = data
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row['formatted_text']
        magnitude = row['Magnitude(ergs)']

        tokens = self.tokeniser(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(magnitude, dtype=torch.float32)
        }


def create_text(row):
    return (f"Earthquake details: Date {row['Date(YYYY/MM/DD)']} at {row['Time(UTC)']}, "
            f"Location: Lat {row['Latitude(deg)']}, Lon {row['Longitude(deg)']}, "
            f"Depth {row['Depth(km)']} km, Magnitude {row['Magnitude(ergs)']} ergs.")


def load_config(file_path):
    """Load configuration file with error handling."""
    try:
        with open(file_path, 'r') as conf:
            configs = yaml.safe_load(conf)
            print(f'Loaded {configs.get("model_name", "unknown")} configurations.')
            return configs
    except (IOError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {e}")
        raise


def save_model(model: PreTrainedModel, tokeniser, output_dir, fold):
    """
    Save the trained model, its weights, and tokeniser
    :param model: Trained model
    :param tokeniser: Tokeniser used
    :param output_dir: Directory to save the model
    :param fold: Current cross-validation fold
    """
    # Create fold-specific directory
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Save model components
    model_save_path = os.path.join(fold_dir, "best_model")
    model.save_pretrained(model_save_path)
    tokeniser.save_pretrained(model_save_path)

    # Save model weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'fold': fold
    }, os.path.join(fold_dir, "model_weights.pth"))

    print(f"Model for fold {fold} saved to {model_save_path}")


def load_model(model_dir, tokeniser):
    """
    Load a saved model
    :param model_dir: Directory containing the saved model
    :param tokeniser: Tokeniser to use
    :return: Loaded model
    """
    # Load base GPT model
    gpt_model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")

    # Create model instance
    model = EarthquakeGPT(gpt_model)

    # Load weights
    weights_path = os.path.join(model_dir, "model_weights.pth")
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def test_model(model, test_df, tokeniser):
    """
    Test the trained model on the test dataset
    :param model: Trained model
    :param test_df: Test dataframe
    :param tokeniser: Tokeniser used
    :return: Tuple of metrics dictionary and predictions
    """
    # Prepare test dataset
    test_df['formatted_text'] = test_df.apply(create_text, axis=1)
    test_data = EarthquakeDataset(test_df, tokeniser)

    # Prepare trainer for prediction
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=8)
    )

    # Make predictions
    results = trainer.predict(test_data)
    pred = results.predictions.squeeze(-1)

    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(test_df["Magnitude(ergs)"], pred),
        'mse': mean_squared_error(test_df["Magnitude(ergs)"], pred),
        'rmse': np.sqrt(mean_squared_error(test_df["Magnitude(ergs)"], pred)),
        'mape': mean_absolute_percentage_error(test_df["Magnitude(ergs)"], pred),
        'r2': r2_score(test_df["Magnitude(ergs)"], pred)
    }

    return metrics, pred


def run_cross_validation(configs, full_df, test_df, tokeniser, n_splits = 5):
    """
    Perform K-Fold cross-validation with model saving
    :param configs: Configuration dictionary
    :param full_df: Full training dataframe
    :param test_df: Separate test dataframe
    :param tokeniser: Tokeniser to use
    :param n_splits: Number of cross-validation splits
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results for each fold
    cv_results = {
        'mae': [], 'mse': [], 'rmse': [],
        'mape': [], 'r2': []
    }

    # Store best model for final testing
    best_model = None
    best_cv_metric = float('inf')
    best_model_dir = None

    # Ensure output directory exists
    output_dir = configs['settings'].get('out_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df), 1):
        print(f"\n{'=' * 20} Fold {fold} {'=' * 20}")

        train_df = full_df.iloc[train_idx]
        val_df = full_df.iloc[val_idx]

        # Preprocess data
        train_df['formatted_text'] = train_df.apply(create_text, axis=1)
        val_df['formatted_text'] = val_df.apply(create_text, axis=1)

        # Prepare datasets
        train_data = EarthquakeDataset(train_df, tokeniser)
        val_data = EarthquakeDataset(val_df, tokeniser)

        # Model and training setup
        gpt_model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
        model = EarthquakeGPT(gpt_model)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/fold_{fold}",
            learning_rate=float(configs['settings']['lr']),
            per_device_train_batch_size=int(configs['settings']['per_device_train_batch_size']),
            num_train_epochs=int(configs['settings']['num_epoch']),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
        )

        # Train
        trainer.train()

        # Save the current model
        save_model(model, tokeniser, output_dir, fold)

        # Predict on validation set
        results = trainer.predict(val_data)
        val_pred = results.predictions.squeeze(-1)

        # Calculate metrics
        fold_metrics = {
            'mae': mean_absolute_error(val_df["Magnitude(ergs)"], val_pred),
            'mse': mean_squared_error(val_df["Magnitude(ergs)"], val_pred),
            'rmse': np.sqrt(mean_squared_error(val_df["Magnitude(ergs)"], val_pred)),
            'mape': mean_absolute_percentage_error(val_df["Magnitude(ergs)"], val_pred),
            'r2': r2_score(val_df["Magnitude(ergs)"], val_pred)
        }

        # Store and print fold results
        for metric, value in fold_metrics.items():
            cv_results[metric].append(value)
            print(f"{metric.upper()}: {value:.4f}")

        # Track best model (using RMSE as metric)
        if fold_metrics['rmse'] < best_cv_metric:
            best_cv_metric = fold_metrics['rmse']
            best_model_dir = os.path.join(output_dir, f"fold_{fold}")

    # Print cross-validation summary
    print("\n" + "=" * 50)
    print("Cross-Validation Summary")
    print("=" * 50)
    for metric, values in cv_results.items():
        print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Load the best model
    best_model = load_model(best_model_dir, tokeniser)
    best_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Test the best model on the test set
    print("\n" + "=" * 50)
    print("Test Set Evaluation")
    print("=" * 50)
    test_metrics, test_predictions = test_model(best_model, test_df, tokeniser)

    # Print test metrics
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # Save testing results
    test_results_df = test_df.copy()
    test_results_df['Predicted_Magnitude'] = test_predictions
    test_results_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)

    # Print best model directory
    print(f"\nBest model saved in: {best_model_dir}")


def main():
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, type=str, help="Path to configuration file")
    ap.add_argument("-m", "--mode", required=False, default="train", type=str, help="The mode to run the model: 'train' or 'test'")
    opts = vars(ap.parse_args())

    # Variables
    mode = opts['mode']

    # Disable external loggers
    os.environ["WANDB_DISABLED"] = "true"

    # Load configurations
    configs = load_config(opts['config'])

    # Prepare tokenizer
    tokeniser = GPT2Tokenizer.from_pretrained("gpt2")
    tokeniser.pad_token = tokeniser.eos_token

    # Load full training dataset and test dataset
    training_df = pd.read_csv(configs['settings']['train_data_path'])
    test_df = pd.read_csv(configs['settings']['test_data_path'])

    if mode == "train":
        # Run cross-validation and testing mode
        run_cross_validation(configs, training_df, test_df, tokeniser)

    elif mode == "test":
        # Run testing mode
        test_gpt2 = load_model(configs['settings']['load_weights'], tokeniser)
        results = test_model(test_gpt2, test_df, tokeniser)

    else:
        print(f"ERROR: '{mode}' is not a valid option. Choose 'train' or 'test'")
        exit()


if __name__ == "__main__":
    main()
