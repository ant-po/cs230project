"""
TODO: write proper description here
"""

import argparse
import datetime
import os
import time
import fix_yahoo_finance as fix
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
from model.utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--data_params', default='data/data_params', help="Directory with the params.json")
parser.add_argument('--data_dir', default='data/raw_data', help="Directory with the raw price data")
parser.add_argument('--output_dir', default='data/processed_data', help="Where to write the new data")


def fetchData(dataCodes, startDate, endDate, output_folder):
    """
    Gets historical stock data of given tickers between dates
    :param dataCode: security (securities) whose data is to fetched
    :type dataCode: string or list of strings
    :param startDate: start date
    :type startDate: string of date "YYYY-mm-dd"
    :param endDate: end date
    :type endDate: string of date "YYYY-mm-dd"
    :return: saves data in a csv file with the timestamps
    """

    fix.pdr_override()
    data = {}
    # for code in dataCodes:
    i = 1
    try:
        all_data = pdr.get_data_yahoo(dataCodes, startDate, endDate)
    except ValueError:
        print("ValueError, trying again")
        i += 1
        if i < 5:
            time.sleep(10)
            fetchData(dataCodes, startDate, endDate)
        else:
            print("Tried 5 times, Yahoo error. Trying after 2 minutes")
            time.sleep(120)
            fetchData(dataCodes, startDate, endDate)
    all_data = all_data.fillna(method="ffill")
    data = all_data["Adj Close"]
    # output the results in a csv file
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
    filename = output_folder + "/raw_data_" + time_stamp + ".csv"
    data.to_csv(filename)
    print("Data has been saved in a CSV format in ", filename)
    return data


def getExistingData(filename):
    """Find existing data file "filename" and extract the data"""
    data = pd.read_csv(filename)
    return data


def saveDataToCsv(x_train, y_train, x_test, y_test, output_folder):
    """Import the training/test data from DataFrame to CSV files"""
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
    output_folder = output_folder + "/data_set_" + time_stamp
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pd.DataFrame(x_train).transpose().to_csv(output_folder + "/x_train_data.csv")
    pd.DataFrame(y_train).transpose().to_csv(output_folder + "/y_train_data.csv")
    pd.DataFrame(x_test).transpose().to_csv(output_folder + "/x_test_data.csv")
    pd.DataFrame(y_test).transpose().to_csv(output_folder + "/y_test_data.csv")
    print("Data has been saved in a CSV format in ", output_folder)


def readDataFromCsv(output_folder):
    """Import the training/test data from CSV files to Numpy arrays"""
    x_train = np.array(pd.read_csv(output_folder+"/x_train_data.csv", index_col=0))
    y_train = np.array(pd.read_csv(output_folder + "/y_train_data.csv", index_col=0))
    x_test = np.array(pd.read_csv(output_folder+"/x_test_data.csv", index_col=0))
    y_test = np.array(pd.read_csv(output_folder + "/y_test_data.csv", index_col=0))
    return x_train, y_train[:, 1], x_test, y_test[:, 1]


def rankToLabel(rank):
    """Generate a label array of size [rank.size**2,1] from the rank
    e.g. rank = [1,2] --> label = [1, 0, 0, 1]"""
    label = np.zeros([rank.size**2, 1])
    num_assets = rank.size
    counter = 0
    for elem in rank:
        counter += 1
        label[(counter-1)*num_assets+int(elem)-1, 0] = 1
    return label


def labelToRank(label):
    """Generate a rank array of size [sqrt(label.size),1] from the label
        e.g. label = [1, 0, 0, 1] --> ranking = [1,2]"""
    rank = np.zeros([int(np.sqrt(label.size)), 1])
    num_assets = rank.size
    for i in range(1, label.size+1):
        if not(i%num_assets):
            rank[int(i/num_assets)-1, 0] = list(label[i-num_assets:i, 0]).index(1)+1
    return rank


def getXYSet(data, look_back, invest_horizon):
    """Slice the data to create training X,Y examples of dimension (invest_horizon, ?) """
    # TODO: code up several alternatives here (see notes)

    # Case 1 - x_set: [invest_horizon * #assets, #examples], y_set: [#assets * #assets, #examples]
    time_series = data.values
    x_set = np.empty([time_series.shape[1]*look_back, 1])
    # y_set = np.empty([time_series.shape[1]**2, 1])
    y_set = np.empty([2, 1])
    for row in range(look_back, time_series.shape[0]-invest_horizon):
        x_back = np.nan_to_num(time_series[row-look_back:row])
        x_set = np.append(x_set, x_back.reshape([x_back.size, 1], order='F'), axis=1)
        x_forw = np.nan_to_num(time_series[row+1:row+invest_horizon])
        y_temp = pd.DataFrame(np.sum(x_forw, axis=0)).rank(axis=0, method='first', ascending=False)
        # y_set = np.append(y_set, rankToLabel(y_temp.values), axis=1)
        y_set = np.append(y_set, np.array([[0], [list(y_temp.values).index(1)]]), axis=1)
    return x_set[:, 1:], y_set[:, 1:]


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_data')
    test_data_dir = os.path.join(args.data_dir, 'test_data')

    # Read data params config
    json_path = os.path.join(args.data_params, 'data_params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    data_params = Params(json_path)

    # Fetch data
    hist_prices = fetchData(data_params.dataCodes, data_params.startDate, data_params.endDate, args.data_dir)

    # Convert prices to log returns
    hist_returns = hist_prices.pct_change(1)

    # Generate train data sets
    train_returns = hist_returns.iloc[:int(data_params.train_prct*hist_returns.shape[0]), :]
    x_train, y_train = getXYSet(train_returns, data_params.look_back, data_params.invest_horizon)

    # Generate dev/test sets
    test_returns = hist_returns.iloc[int(data_params.train_prct*hist_returns.shape[0])+1:int((data_params.train_prct+data_params.test_prct)*hist_returns.shape[0]), :]
    x_test, y_test = getXYSet(test_returns, data_params.look_back, data_params.invest_horizon)

    # Output the results to csv
    saveDataToCsv(x_train, y_train, x_test, y_test, args.output_dir)
    print("Done building dataset. Ready to train the model now!")
