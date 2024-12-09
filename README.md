# EarthquakeGPT

**_Abstract_**: The design of earthquake-resistant structures is a multidisciplinary field where advancements in Machine 
Learning (ML) and Artificial Intelligence (AI) are driving significant innovations. Peak Ground Acceleration (PGA), 
commonly referred to as earthquake magnitude, is a critical parameter for designing structures capable of withstanding 
seismic forces. This study introduces a regressive GPT-2 model specifically designed to predict earthquake magnitudes in 
California, United States, based on historical seismic data. The model leverages the natural language processing 
capabilities of GPT-2, repurposed for numerical regression tasks. Our results demonstrate the efficacy of the regressive 
GPT-2 model, achieving a Mean Absolute Error (MAE) of 0.0228, Mean Squared Error (MSE) of 0.0016, Root Mean Squared 
Error (RMSE) of 0.0399, Mean Absolute Percentage Error (MAPE) of 0.0061, and a coefficient of determination (R2) of 
0.9917. These metrics highlight the model’s exceptional predictive accuracy and potential as a robust tool for seismic 
risk assessment. By providing precise magnitude predictions, this work contributes to the development of safer, 
earthquake-resilient infrastructure and highlights the potential of AI-driven approaches in geophysics.

## Initial Results

### Cross-Validation Summary
MAE: 0.0249 ± 0.0024  
MSE: 0.0020 ± 0.0007  
RMSE: 0.0444 ± 0.0081  
MAPE: 0.0065 ± 0.0005  
R2: 0.9895 ± 0.0034

### Comparative Analysis:

|      Models       | MAE  &darr; | MSE &darr; | RMSE &darr; | MAPE &darr; | R2 &uarr; |
|:-----------------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|
| Linear Regression |     ---     |   0.182    |     ---     |     ---     | 0.000067  |
|        SVM        |     ---     |   7.210    |     ---     |     ---     |  -38.618  |
|  Random Fortest   |     ---     |   0.143    |     ---     |     ---     |   0.212   |
|        KAN        |             |            |             |             |           |
|        GRU        |             |            |             |             |           |  
|       LSTM        |             |            |             |             |           |
|    Transformer    |    0.299    |   0.218    |    0.467    |    0.079    |   0.250   |
| TimeMixer (SOTA)  |             |            |             |             |           |
|  PatchTST (SOTA)  |             |            |             |             |           |
|  EQGPT 2 (Ours)   |   0.0228    |   0.0016   |   0.0399    |   0.0061    |  0.9917   |

> The results presented for the EQGPT2 model represent the outcomes of a five-fold cross-validation process.
---

## Dataset
To evaluate and train our model and the comparative models, we used the [SOCR Earthquake Data](http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Earthquakes_Over3.html), which can be found in the `Data/` directory. We split the dataset into 80%(14425 samples) training and 20% (3607 samples) testing sets.

---
## How to run our models:

#### 1. Install Environment and Dependencies
To install the environment, use the following command:
```commandline
conda env create -f environment.yml
```

#### Inferencing: EarthquakeGPT
We provide two versions of our model: single trained and five-fold cross-validation trained. Both of the models can be 
run using the commands in this document. To run our model in inferencing model on our dataset, run the following command:
```commandline
cd code
python Main_model/gpt_2.py -c Main_model/configs/gpt_2_configs.yaml -m test
```
> NOTE: Make sure the `load_weights` path in the configuration file is set to where you have stored you pre-trained weights. 

#### Inferencing: Comparative Methods
We compared our approach to the following methods:

- Gated Recurrent Unit ([GRU](https://nixtlaverse.nixtla.io/neuralforecast/models.gru.html)) - `gru.py`
- Kolmogorov-Arnold Networks ([KAN](https://nixtlaverse.nixtla.io/neuralforecast/models.kan.html)) - `kan.py`
- Long Short-Term Memory Recurrent Neural Network ([LSTM](https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html)) - `lstm.py`
- Vanilla Transformer ([Transformer](https://nixtlaverse.nixtla.io/neuralforecast/models.vanillatransformer.html)) - `transformer.py`
- ([TimeMixer](https://nixtlaverse.nixtla.io/neuralforecast/models.timemixer.html)) - `time_mixer.py`
- ([PatchTST](https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html)) - `patchtst.py`

You can run any of these models using the following template: 
```commandline
python code/Comparative_models/<model_name>.py -t Data/train_earthquake_data.csv -v Data/test_earthquake_data.csv -m test
```

#### Training: Our Dataset
To train our model on our dataset, you can run the following code:
```commandline
cd code
python Main_model/gpt_2.py -c Main_model/configs/gpt_2_configs.yaml -m train
```
You can tweak the configurations of the model in the `code/Main_model/configs/gpt_2_config.yaml`. Here, you can change 
where to store the weights and your output directory.

#### Training: Your dataset
1. Ensure your dataset is a structured table (e.g., a CSV or DataFrame) with the following columns:
   - Date(YYYY/MM/DD): Date of the earthquake in the format YYYY/MM/DD.
   - Time(UTC): Time of the earthquake in Coordinated Universal Time (UTC).
   - Latitude(deg): Latitude of the earthquake's epicentre in degrees.
   - Longitude(deg): Longitude of the earthquake's epicentre in degrees.
   - Depth(km): Depth of the earthquake in kilometres.
   - Magnitude(ergs): Magnitude of the earthquake in ergs (used as the target variable).

2. You will also have to change the `train_data_path` and `test_data_path` in the configuration file in the `code/Main_Model/config/gpt_2_config.yaml`.

3. Then you can use the training command provided above:
```commandline
python Main_model/gpt_2.py -c Main_model/configs/gpt_2_configs.yaml -m train
```
---

## Acknowledgement

In this project we utilised:

- [Nixtla](https://nixtlaverse.nixtla.io/) Library:
```
@misc{olivares2022library_neuralforecast,
    author={Kin G. Olivares and
            Cristian Challú and
            Federico Garza and
            Max Mergenthaler Canseco and
            Artur Dubrawski},
    title = {{NeuralForecast}: User friendly state-of-the-art neural forecasting models.},
    year={2022},
    howpublished={{PyCon} Salt Lake City, Utah, US 2022},
    url={https://github.com/Nixtla/neuralforecast}
}
```
- [OpenAI](https://openai.com/) GPT2