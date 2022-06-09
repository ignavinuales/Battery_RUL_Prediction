# Battery_RUL_Prediction
The project objective is to predict a battery's remaining useful life (RUL) by developing a feedforward and a Long-Short-Time-Memory (LSTM) 
neural network, using features based on voltage, current, and time extracted from public battery condition monitoring datasets.

The public datasets can be found here: https://www.batteryarchive.org/list.html
The databases selected are from HNEI source. The .csv files are the time series named 'HNEI_18650_NMC_LCO_25C_0-100_0.5/1.5C_'
14 of them are selected. 

# DATA PREPROCESSING
The publicly available databases for battery life cycles do not provide ready-to-use data for our research.
Although they contain different variables such as voltage (V), current (A), time (S), discharge and charge capacity (Ah), 
and charge and discharge energy (Wh), not all of them can be used on our model. Our objective is to use only voltage, current, and time as inputs. 
Nonetheless, employing these variables directly as inputs is not feasible since they provide meaningless information and are not sufficient 
to create a model. Therefore, they need to be treated and manipulated as a base to develop new features which the neural network will train with. 

In summary, seven features are created from the source datasets using voltage, current, and time. The idea is to use those features to predict the RUL
of the battery. The features are summarized in figures named 'Voltage Discharging Cycle', 'Voltage Discharging Cycle', and 'Current Charing Cycle'.

# MODULES:
preprocess_data.py: preprocesses data, creates the features and appends them to a new dataset. 
join_dataframes.py: it concatenates the 14 dataframes (one for each of the 14 source dataframes) createed by preprocess_data.py.
FeedForward_NN.py: Feedforward neural network.
LSTM_NN.py: LSTM neural network.

# AVAILABLE DATASETS:
The datasets created by preprocess_data.py and the joined dataset can be accessed.
However, due to the size of the HNEI source datasets, these are not available in the repository. 
