# EarthquakeGPT

Abstract: 


### Comparative Analysis:

|      Models      | MAE | MSE | RMSE | MAPE | $R^2$ |
|:----------------:|:---:|:---:|:----:|:----:|:-----:|
|       KAN        |     |     |      |      |       |
|       GRU        |     |     |      |      |       |  
|       LSTM       |     |     |      |      |       |
|   Transformer    |     |     |      |      |       |
| TimeMixer (SOTA) |     |     |      |      |       |
| PatchTST (SOTA)  |     |     |      |      |       |
|   GPT 2 (Ours)   |     |     |      |      |       |

### Dataset
To evaluate and train our model and the comparative models, we used the [SOCR Earthquake Data](http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Earthquakes_Over3.html), which can be found in the `Data/` directory. We split the dataset into 80%(14425 samples) training and 20% (3607 samples) testing sets.

### How to run our models:

#### 1. Install Environment and Dependencies
To install the environment, use the following command:
```commandline
conda env create -f environment.yml
```

#### Inferencing: EarthquakeGPT
To run our model in inferencing model on our dataset, run the following command:
```commandline
cd code
python Main_model/gpt_2.py -c Main_model/configs/gpt_2_configs.yaml -m test
```
> NOTE: Make sure the `load_weights` path in the configuration file is set to where you have stored you pre-trained weights. 

#### Inferencing: Comparative Methods

#### Training: Our Dataset
To train our model on our dataset, you can run the following code:
```commandline
cd code
python Main_model/gpt_2.py -c Main_model/configs/gpt_2_configs.yaml -m train
```
You can tweak the configurations of the model in the `code/Main_model/configs/gpt_2_config.yaml`. Here, you can change 
where to store the weights and your output directory.

#### Training: Your dataset


### Credit

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