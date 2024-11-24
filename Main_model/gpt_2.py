"""
This module is the training (fine-tuning) and inferencing code for GPT 2 model.

Author: Deja S.
Version: 1.0.6
Created: 30-10-2024
Edited: 23-11-2024
"""

import argparse
import os

import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, \
    mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2Config, GPT2Model, PreTrainedModel


class EarthquakeGPT(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, gpt_model):
        super().__init__(gpt_model.config)
        self.gpt = gpt_model
        self.regression_head = torch.nn.Linear(gpt_model.config.n_embd, 1)
        self.criterion = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)[0]
        pred_mag = self.regression_head(gpt_output[:, -1, :])

        # Always calculate loss even if labels is None (will be zero in this case)
        loss = torch.tensor(0.0, device=pred_mag.device)
        if labels is not None:
            loss = self.criterion(pred_mag.squeeze(-1), labels)

        return {
            "loss": loss,
            "logits": pred_mag
        }


class EarthquakeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


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
    return (f"On {row['Date(YYYY/MM/DD)']} at {row['Time(UTC)']}, "
            f"an earthquake occurred at Latitude {row['Latitude(deg)']}, Longitude {row['Longitude(deg)']}, "
            f"with Depth {row['Depth(km)']} km and Magnitude {row['Magnitude(ergs)']}.")


def load_config(file_path):
    """
    Load configuration file.
    :param file_path: The string file path.
    :return: Any
    """
    with open(file_path, 'r') as conf:
        configs = yaml.load(conf, Loader=yaml.FullLoader)
        print(f'Loaded {configs["model_name"]} configurations.')
        return configs


def main():
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, type=str, help="The path to the config file.")
    ap.add_argument("-m", "--mode", required=False, default="train", type=str,
                    help="The mode to run the model: 'train' or 'test'")
    opts = vars(ap.parse_args())

    # Disable Logger
    os.environ["WANDB_DISABLED"] = "true"

    # Required Variables
    config_file_path = opts['config']
    mode = opts['mode']

    # Check if config path exist
    if not os.path.exists(config_file_path):
        print(f"ERROR: '{config_file_path}' can not be found.")
        exit()

    configs = load_config(config_file_path)
    train_dir = os.path.join(configs['settings']['out_dir'], 'train')
    test_dir = os.path.join(configs['settings']['out_dir'], 'test')

    # Create output directories
    if not os.path.exists(configs['settings']['out_dir']):
        os.mkdir(configs['settings']['out_dir'])

    if mode == "train":
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

    if mode == "test":
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

    # Load GPT-2 tokeniser
    tokeniser = GPT2Tokenizer.from_pretrained("gpt2")
    tokeniser.pad_token = tokeniser.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load in the datasets
    temp_train_df = pd.read_csv(configs['settings']['train_data_path'])
    test_df = pd.read_csv(configs['settings']['test_data_path'])

    # Split data for validation set
    train_df, valid_df = train_test_split(temp_train_df, test_size=0.2)

    # Preprocess data
    train_df['formatted_text'] = train_df.apply(create_text, axis=1)
    valid_df['formatted_text'] = valid_df.apply(create_text, axis=1)
    test_df['formatted_text'] = test_df.apply(create_text, axis=1)

    # Training or Inference step
    if mode == "train":
        # Model Configurations
        gpt_model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
        model = EarthquakeGPT(gpt_model)
        model.to(device)

        # Configure data
        train_data = EarthquakeDataset(train_df, tokeniser)
        valid_data = EarthquakeDataset(valid_df, tokeniser)

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=configs['settings']['out_dir'],
            eval_strategy=configs['settings']['evaluation_strategy'],
            learning_rate=float(configs['settings']['lr']),
            per_device_train_batch_size=int(configs['settings']['per_device_train_batch_size']),
            per_device_eval_batch_size=int(configs['settings']['per_device_eval_batch_size']),
            num_train_epochs=int(configs['settings']['num_epoch']),
            warmup_steps=int(configs['settings']['warmup_steps']),
            weight_decay=float(configs['settings']['weight_decay']),
            remove_unused_columns=True,
            gradient_accumulation_steps=int(configs['settings']['gradient_accumulation_steps']),
            logging_steps=int(configs['settings']['logging_steps']),
            save_strategy=configs['settings']['save_strategy'],
            disable_tqdm=False,
            load_best_model_at_end=True,
        )

        # Trainer Arguments
        trainer = EarthquakeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            tokenizer=tokeniser,
        )

        # Train model
        trainer.train()

        # Save Model
        model_path = f"{train_dir}/fine-tuned-gpt2"
        model.save_pretrained(model_path)
        tokeniser.save_pretrained(model_path)

    elif mode == "test":
        print("Evaluating Model ...")

        # Model configurations
        model = EarthquakeGPT.from_pretrained(configs['settings']['load_weights'])
        model.eval()
        model.to(device)

        # Configure data
        test_data = EarthquakeDataset(test_df, tokeniser)

        # Model Prediction
        trainer = EarthquakeTrainer(
            model=model,
            args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=8)
        )
        results = trainer.predict(test_data)
        pred = results.predictions.squeeze(-1)

        # Calculate the metrics
        mae = mean_absolute_error(test_df["Magnitude(ergs)"], pred)
        mse = mean_squared_error(test_df["Magnitude(ergs)"], pred)
        rmse = root_mean_squared_error(test_df["Magnitude(ergs)"], pred)
        mape = mean_absolute_percentage_error(test_df["Magnitude(ergs)"], pred)
        r2 = r2_score(test_df["Magnitude(ergs)"], pred)

        print()
        print("Evaluation Results")
        print("=" * 80)
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAPE: {mape:.3f}")
        print(f"R2: {r2:.3f}")

    else:
        print(f"ERROR: '{mode}' is not a valid option. Choose 'train' or 'test'")
        exit()


if __name__ == "__main__":
    main()