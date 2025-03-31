import os
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import Constants
from utils.metrics import metric

cross_list = Constants().cross_list


def read_results(results_path, station_results_path):
    file_path = os.path.join(results_path, station_results_path)
    pred_npy = os.path.join(file_path, "pred.npy")
    true_npy = os.path.join(file_path, "true.npy")
    metrics_npy = os.path.join(file_path, "metrics.npy")

    pred = np.load(pred_npy)
    true = np.load(true_npy)
    metrics = np.load(metrics_npy)  # metric = np.array([mae, mse, rmse, mape, mspe]
    return pred, true, metrics


def get_cross_station_results(station, model_name, results_path):
    station2_list = [station2 for (station1, station2) in cross_list]
    if station not in station2_list:
        station_results_path = f"{model_name}_MeKong_{station.replace(' ', '')}__ftM_sl3_ll0_pl1_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_"
        pred, true, metrics = read_results(results_path, station_results_path)
    else:
        pred_list = []
        # predict from itself
        station_results_path = f"{model_name}_MeKong_{station.replace(' ', '')}__ftM_sl3_ll0_pl1_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_"
        pred, true, metrics = read_results(results_path, station_results_path)
        pred_list.append(pred)
        plt.plot(pred[:, 0, 0])
        plt.plot(true[:, 0, 0])
        plt.show()
        # predict from other stations
        for (station1, station2) in cross_list:
            if station == station2:
                station_results_path = f"{model_name}_MeKong_Cross_{station1.replace(' ', '')}_{station2.replace(' ', '')}_ftM_sl3_ll0_pl1_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_"
                pred, true, metrics = read_results(results_path, station_results_path)
                pred_list.append(pred)
                plt.plot(pred[:, 0, 0])
                plt.plot(true[:, 0, 0])
                plt.show()
        pred = np.mean(pred_list, axis=0)
        metrics = metric(pred, true)
    return pred, true, metrics


results_folder = '../results/iTransformer'
model_name = 'iTransformer'

station = 'Ban Don'


# pred_npy = os.path.join(station_path, 'pred.npy')
# true_npy = os.path.join(station_path, 'true.npy')
pred, true, metrics = get_cross_station_results(station, model_name, results_folder)
# exit()
pred = pred[:, 0, 0]
true = true[:, 0, 0]
rmse = metrics[2]
print('rmse:', rmse)
plt.plot(pred)
plt.plot(true)
plt.show()



