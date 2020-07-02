import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


# Function to write the json file
def write_dicts_to_file(path, data):
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))


# The first example dataset can be downloaded from the link below
# dataset: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
data = pd.read_csv('LD2011_2014.txt',
                   sep=";",
                   index_col=0,
                   parse_dates=True,
                   decimal=',')

# Aggregate values in a 2 hour window
freq = '2H'
data_kw = data.resample(freq).sum() / 8
timeseries = []
num_timeseries = data_kw.shape[1]

# In the interest keep the file size small, create an array of length 1000
for i in range(num_timeseries):
    timeseries.append(np.trim_zeros(data_kw.iloc[10000:11000,i], trim='f'))

"""
# You can plot some of the series to examine them visually
fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=True)
axx = axs.ravel()
num_series_visualize = 10
for i in range(0, num_series_visualize):
    timeseries[i].loc["2013-04-13":"2013-05-13"].plot(ax=axx[i])
    axx[i].set_xlabel("date")    
    axx[i].set_ylabel("kW consumption")   
    axx[i].grid(which='minor', axis='x')
"""

# Create the test and train file
start_dataset = pd.Timestamp("2013-04-13 00:00:00", freq=freq)
end_training = pd.Timestamp("2013-06-13 00:00:00", freq=freq)
end_testing = pd.Timestamp("2013-07-13 00:00:00", freq=freq)

training_data = [
    {
        "start": str(start_dataset),
        "target": ts[start_dataset:end_training - 1].tolist()
    }
    for ts in timeseries
]

test_data = [
    {
        "start": str(end_training),
        "target": ts[end_training: end_testing].tolist()
    }
    for ts in timeseries
]
len(test_data[0]['target'])

write_dicts_to_file("LD_train.json", training_data)
write_dicts_to_file("LD_test.json", test_data)

# A second example dataset again from UCI, link for download of dataset can be found below
# https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
data_individual = pd.read_csv('household_power_consumption.txt',
                              sep=";",
                              index_col=0,
                              parse_dates=True,
                              decimal=',',
                              na_values=['?'])

# Convert datatype from object to float so that aggregation can be applied
data_individual['Sub_metering_1'] = data_individual['Sub_metering_1'].astype('float')
data_individual['Sub_metering_2'] = data_individual['Sub_metering_2'].astype('float')
data_individual['Sub_metering_3'] = data_individual['Sub_metering_3'].astype('float')

# Aggregate for one day
data_kw_household = data_individual.resample('1D').sum()

# Aggregate readings from the 3 meters
data_kw_household['total_reading'] = data_kw_household['Sub_metering_1'] + \
                                     data_kw_household['Sub_metering_2'] + \
                                     data_kw_household['Sub_metering_3']


"""
# Visualize the dataset for each year
fig, axs = plt.subplots(3, 1, figsize=(20, 20))
axx = axs.ravel()
for i in range(0, 3):
    data_kw_household['total_reading'].loc[years[i]].plot(ax=axx[i])
    axx[i].set_xlabel("date")    
    axx[i].set_ylabel("kW consumption")   
    axx[i].grid(which='minor', axis='x')
"""

# Write the test and train datasets
years = ['2007','2008', '2009']
file = open("individual_household_train.json", "a")
for year in years:
    # One JSON sample per line
    line = "\"start\":\"{} 00:00:00\",\"target\":{}".format(year, data_kw_household[year]['total_reading'].tolist())
    file.write('{'+line+'}\n')
file.close()

years = ['2010']
file = open("individual_household_test.json", "a")
for year in years:
    # One JSON sample per line
    line = "\"start\":\"{} 00:00:00\",\"target\":{}".format(year, data_kw_household[year]['total_reading'].tolist())
    file.write('{'+line+'}\n')
file.close()