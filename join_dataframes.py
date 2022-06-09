import pandas as pd
import os

# JOINS THE 14 DATABASES INTO ONE:
# The source databases are named as 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a_timeseries.csv'.
# In all 14 databases the only change to the name is the letter before '_timeseries.csv'
# Ranging from 'a' to 't'


def import_df(file: str):
    """
    Get the dataframe by stating the letter ('a', 'b', 'c', etc.) from the name of the .csv file
    Parameters
    ----------
    file: letters ('a', 'b', 'c', etc.)

    Returns
    -------
    The dataframe df_features from the main.py
    """
    name_file = 'HNEI_' + file + '_features.csv'
    df_features = pd.read_csv(os.getcwd() + '/Datasets/HNEI_Processed/' + name_file)
    
    return df_features


final_df = pd.DataFrame()  # Creating a new dataframe
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'j', 'l', 'n', 'o', 'p', 's', 't']
for i in files:
    df_features = import_df(i)
    final_df = final_df.append(df_features)  # Appending each dataframe to the final dataframe
    
final_df.drop('Unnamed: 0', axis=1, inplace=True)  # Dropping one unuseful column

# Saving the final dataframe
name_database = 'Final Database.csv'
final_df.to_csv(os.getcwd() + '/Datasets/HNEI_Processed/' + name_database)


# final_df.sort_values(by='Cycle_Index', inplace=True)
# final_df.drop('Unnamed: 0', axis=1, inplace=True)
# name_database = 'Final Database.csv'
# final_df.to_csv(os.getcwd() + '/Datasets/HNEI_Processed/' + name_database)
