import pandas as pd
import os


def closest_values(data, target_value: float, cycle: int, phase: bool):
    """
    Approximating the value of ´x´ for a given value of ´y´
    of a function f(x)=y, and get the time of 'x'

    Parameters
    ----------
    data : dataframe.
    target_value : the value of y (desired voltage value).
    cycle : cycle of the battery.
    phase : bool
        1 for discharging phase, 0 for charging phase.

    Returns
    -------
    feature : the time (s) of x in the given cycle

    """
    
    if phase == 0:
        df_filt = data[data['Current (A)'] > 0]  # Selecting charging phase
    
    elif phase == 1:
        df_filt = data[data['Current (A)'] < 0]  # Selecting discharging phase

    df_filt = df_filt[df_filt['Cycle_Index'] == cycle]  # Selects battery cycle
    
    # Look for the closest value to target_value
    a_list = list(df_filt['Voltage (V)'])
    absolute_difference_function = lambda list_value: abs(list_value - target_value)
    
    closest_value = min(a_list, key=absolute_difference_function)   # Closest value to target_value
    ind_1 = a_list.index(closest_value)  # Get index of the value
    time_closest_value = df_filt.reset_index()['Test_Time (s)'][ind_1]  # Time (s) for target_value
    
    cycle_start = df_filt['Test_Time (s)'].min()  # Time when the charging cycle starts.
    
    if closest_value == target_value:
        feature = time_closest_value - cycle_start
        
    else:   # Finding the value between two points
        b_list = a_list.copy()
        b_list.remove(closest_value)
        second_closest_value = min(b_list, key=absolute_difference_function)  # Second closest value to target value
        ind_2 = a_list.index(second_closest_value)  # Get index of the value
        time_second_closest_value = df_filt.reset_index()['Test_Time (s)'][ind_2]  # Time (s) for that value
        
        y1 = time_closest_value
        y2 = time_second_closest_value
        x1 = closest_value
        x2 = second_closest_value
        x = target_value
  
        if closest_value < second_closest_value:
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)  # Linear approximation
            
        else:
            y = y2 + ((x - x2) / (x1 - x2)) * (y1 - y2)  # Linear approximation
                  
        feature = y - cycle_start
                
    return feature


# ====================================================================================
# IMPORT THE DATASET:
    
# Name of the file we want to import (add .csv at the end)
name_file = 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a_timeseries.csv'
raw_df = pd.read_csv(os.getcwd() + '/Datasets/Raw Datasets/' + name_file)
df = raw_df.copy(deep=True)

# FINDING OUT HOW MANY CYCLES STARTS WITH charging OR DISCHARGING PHASE
list_discharging = []
first_charging = 0
first_discharging = 0
last_cycle = int(df['Cycle_Index'].max())
for cycle in range(1, last_cycle):
    try:
        df_test = df[df['Cycle_Index'] == cycle]  # Selecting the cycle
        df_test = df_test[df_test['Current (A)'] != 0]  # Selecting current (A) values not equal to zero
        df_test = df_test['Current (A)']
        first_value = df_test.reset_index()['Current (A)'][0]

        # If the first non-zero current (A) value of a new cycle is positive, then charging takes place first
        if first_value > 0:
            first_charging += 1

        # If the first non-zero current (A) value of a new cycle is negative, then discharging takes place first
        else:
            first_discharging += 1
            list_discharging.append(cycle)
    except:
        pass
        
print(f'{first_discharging} starts with discharging')
print(f'{first_charging} starts with charging')
print(f'The following list contains the cycles which start with discharge phase: {list_discharging}')

# ====================================================================================
# FEATURE 1: 'Discharge Time (s)' is the time that takes the voltage to reach its
# minimun value in one discharge cycle.

# Creating new dataset 'df_features'  which will contain new designed features.
df_features = pd.DataFrame(columns=['Cycle_Index','Discharge Time (s)'])
df_features = df_features.append({'Cycle_Index': '', 'Discharge Time (s)': ''},
                                 ignore_index=True)
    
last_cycle = int(df['Cycle_Index'].max())  # Max number of cycles in the original dataframe 'df'

# Get the discharge time for every cycle in 'df'
for cycle in range(1, last_cycle + 1):
    filt = (df['Current (A)'] < 0) & (df['Cycle_Index'] == cycle)  # Filtering only for discharge phase and cycle

    # Discharge time is the difference between max and min time within the cycle as the time is cumulative in 'df'
    disch_time = round(df[filt]['Test_Time (s)'].max() - df[filt]\
                       ['Test_Time (s)'].min(), 2)

    # Append result to df_features
    df_features = df_features.append({'Cycle_Index': cycle,
                                      'Discharge Time (s)': disch_time}, ignore_index=True)

# ====================================================================================
# FEATURE 2: 'Decrement 3.6-3.4V (s)'
# It represents the time which the voltage takes to drop from 3.6V to 3.4V
# during a discharge cycle.

last_cycle = int(df['Cycle_Index'].max())  # Max number of cycles in the original dataframe 'df'

for cycle in range(1, last_cycle + 1):

    try:
        time_1 = closest_values(df, 3.6, cycle, 1)  # Calling function to approximate the time at 3.6V
        time_2 = closest_values(df, 3.4, cycle, 1)  # Calling function to approximate the time at 3.4V

        feature_2 = time_2 - time_1

        df_features.loc[cycle, 'Decrement 3.6-3.4V (s)'] = feature_2

    except:
        pass

# ====================================================================================
# FEATURE 3: 'Max. Voltage Dischar. (V)'
# It's the initial and maximum voltage in the discharging phase.

last_cycle = int(df_features.shape[0])

for cycle in range(1, last_cycle):
    try:
        df_filt = df[df['Cycle_Index'] == cycle]  # Select cycle
        df_filt = df_filt[df_filt['Current (A)'] < 0]  # Discharge phase
        
        # Calculate feature 3:
        max_voltage = df_filt['Voltage (V)'].max() 
        
        # Append to df_features:
        df_features.loc[cycle, 'Max. Voltage Dischar. (V)'] = round(max_voltage,  6)
        
    except:
        pass

# ====================================================================================
# FEATURE 4: Min. Voltage Charg. (V)
# It's the initial value of Voltage when charging.

last_cycle = int(df_features['Cycle_Index'].shape[0])

for cycle in range(1, last_cycle):
    try:
        df_filt = df[df['Cycle_Index'] == cycle]  # Select cycle
        df_filt = df_filt[df_filt['Current (A)'] > 0]  # Charging phase
        
        # Calculate feature 4:
        min_voltage = df_filt['Voltage (V)'].min() 
        
        # Append in df_features:
        df_features.loc[cycle, 'Min. Voltage Charg. (V)'] = round(min_voltage, 6)
        
    except:
        pass

# ====================================================================================
# FEATURE 5: 'Time at 4.15V (s)'
# It's the time to reach 4.15V in charging phase

last_cycle = int(df['Cycle_Index'].max())
for cycle in range(1, last_cycle + 1):
    
    try:
        feature_5 = closest_values(df, 4.15, cycle, 0)  # Calling function to approx. the time at 4.15V when charging
        df_features.loc[cycle, 'Time at 4.15V (s)'] = feature_5  # Insert feature in the features dataframe
            
    except:
        pass

# ====================================================================================
# FEATURE 6: 'Time constant current (s)'
# It's the time in which the current stays constant at its max. value

last_cycle = int(df_features.shape[0])
for cycle in range(1, last_cycle + 1):
    try:
        df_filt = df[df['Cycle_Index'] == cycle]  # Select cycle
        df_filt = df_filt[df_filt['Current (A)'] > 0] # Select charging phase
        
        max_current = df_filt['Current (A)'].mode()
        max_current -= 0.05
        df_current = df_filt[df_filt['Current (A)'] > max_current[0]]
        
        #Calculate Feature 6:
        feature_6 = df_current['Test_Time (s)'].max() - \
            df_current['Test_Time (s)'].min()
        
        # Append the feature to df_features
        df_features.loc[cycle, 'Time constant current (s)'] = round(feature_6, 2)
                
    except:
        pass
    
# ====================================================================================
# FEATURE 7: ('Charging time (s)')
# It's the total time for charging

last_cycle = int(df['Cycle_Index'].max())
for cycle in range(1, last_cycle + 1):
    try:
        # Total time accumulated (charging + discharging)
        df_filt = df[df['Cycle_Index'] == cycle]
        total_time = float(df_filt['Test_Time (s)'].max())
        
        # Total time of each charging phase
        df_filt = df_filt[df_filt['Current (A)'] > 0]
        charging_time = float(df_filt['Test_Time (s)'].max() \
                              - df_filt['Test_Time (s)'].min())
        
        # Adding values to df_features
        df_features.loc[cycle, 'Total time (s)'] = round(total_time, 2)  # Charging + Discharging
        df_features.loc[cycle, 'Charging time (s)'] = round(charging_time, 2)  # Charging time
        
    except:
        pass

# Difference of times between cycles. Now we also have the total of EACH cycle
df_total_time = df_features['Total time (s)'].diff()
df_features.drop('Total time (s)', axis=1, inplace=True)  # Drop the accumulated times
df_features = df_features.join(df_total_time)  # Append the total time of each cycle

# The total time for the first cycle is missing. We'll add it: 
total_time_cycle_1 = df[df['Cycle_Index'] == 1]['Test_Time (s)'].max()
df_features.loc[1, 'Total time (s)'] = round(total_time_cycle_1, 2)

# ====================================================================================
# DATA CLEANING:

# Dropping first row since it has no values:
df_features.drop(0, axis=0, inplace=True)

# ADDING THE RUL (REMAINING USEFUL LIFETIME) FOR EVERY CYCLE
# RUL = Last cycle - current cycle
df_features['RUL'] = ''
last_cycle = int(df_features.shape[0] + 1)
for cycle in range(1, last_cycle):
    max_cycle = df_features.shape[0]  # Maximum battery cycle
    RUL = int(max_cycle - cycle)  # Calculating RUL
    df_features.loc[cycle, 'RUL'] = RUL  # Adding the value to the dataframe
    
# Checking empty cycles and dropping them from df_features:
empty_cycles = df_features["Discharge Time (s)"].isna().sum()
print(f'There are {empty_cycles} empty cycles without any value at all.')
df_features.dropna(inplace=True)

# ====================================================================================
# Save df_features as a .csv file:

name_database = 'HNEI_a'
df_features.to_csv(os.getcwd() + '/Datasets/HNEI_Processed/' + name_database +
                   '_features.csv')
