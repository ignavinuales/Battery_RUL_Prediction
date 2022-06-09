# Battery RUL prediction using PyTorch
#### Objectve: 
The  objective is twofold:
1. Create new features based on voltage, current, and time. 
2. Predict a battery's remaining useful life (RUL) by developing a feedforward and a Long-Short-Time-Memory (LSTM) neural network using the new features.

#### Source datasets:
The public datasets can be found here: https://www.batteryarchive.org/list.html
14 databases are selected from the HNEI source. The .csv files are the time series named 'HNEI_18650_NMC_LCO_25C_0-100_0.5/1.5C_'.

## Data preprocessing
The publicly available databases for battery life cycles do not provide ready-to-use data for this project. Although they contain different variables such as voltage (V), current (A), time (S), discharge and charge capacity (Ah), and charge and discharge energy (Wh), not all of them can be used for this project. The objective is to use only voltage, current, and time as inputs. Nonetheless, employing these variables directly as inputs is not feasible since they provide meaningless information and are not sufficient to create a model. Therefore, they need to be treated and manipulated as a base to develop new features which the neural network will train with. 

In summary, seven features are created from the source datasets using voltage, current, and time. The idea is to use those features to predict the RUL of the battery. The features are summarized in the figures below:

<img src="https://github.com/ignavinuales/Battery_RUL_Prediction/blob/main/Voltage%20Discharging%20Cycle.png"  width="300" height="300"> <img src="https://github.com/ignavinuales/Battery_RUL_Prediction/blob/main/Voltage%20Charging%20Cycle.png"  width="300" height="300"> <img src="https://github.com/ignavinuales/Battery_RUL_Prediction/blob/main/Current%20Charging%20Cycle.png"  width="300" height="300"> 
## Modules
**preprocess_data.py:** preprocesses data, creates the features and appends them to a new dataset. This code needs to be run for each and one of the 14 source databases.\
**join_dataframes.py:** it concatenates the 14 dataframes (one for each of the 14 source dataframes) createed by preprocess_data.py.\
**FeedForward_NN.py:** Feedforward neural network.\
**LSTM_NN.py:** LSTM neural network.

## Datasets in the repo
The datasets created by preprocess_data.py and the joined dataset can be accessed. However, due to the size of the HNEI source datasets, these are not available in the repository. 
