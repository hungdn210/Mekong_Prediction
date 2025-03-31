import os
import numpy as np
from utils.configs import BasicConfigs
from utils.constants import Constants
from utils.metrics import metric
import matplotlib.pyplot as plt
import copy
import csv

configs = BasicConfigs()


class MakeTable:
    def __init__(self, model, results_path, tables_save_path=None, csv_save_path=None):
        if tables_save_path is None:
            tables_save_path = os.path.join('latex_tables')
        if csv_save_path is None:
            csv_save_path = os.path.join('csv_files')

        self._model = model
        self._results_path = results_path
        self._tables_save_path = tables_save_path
        self._csv_save_path = csv_save_path
        self._configs = configs
        self._all_stations = Constants().all_stations
        self._cross_list = Constants().cross_list
        self._station2_list = [station2 for (station1, station2) in self._cross_list]

        self._latex_table_template = self._create_table_template()

        self._configs.model_default_configs(self._model)

        if not os.path.exists(self._tables_save_path):
            os.makedirs(self._tables_save_path)
        if not os.path.exists(self._csv_save_path):
            os.makedirs(self._csv_save_path)

    def _create_table_template(self):
        latex_table_template = r"""
        \begin{table*}
        \caption{}
        \label{}
        \small
        \centering
        \setlength{\tabcolsep}{1mm}
        \begin{tabular}{@{}c|c|c|c@{}}
        \toprule
        Station & RMSE & MSE & MAE \\ \midrule
        """

        for station in self._all_stations:
            latex_table_template += rf"""
            {station} & {''} & {''} & {''} \\ \midrule"""

        latex_table_template += rf"""
            Avg & {''} & {''} & {''} \\ \midrule"""
        latex_table_template += r"""
        \end{tabular}
        \end{table*}
        """
        return latex_table_template

    def _get_station_path1(self, station):
        return (f"{model_name}_"
                f"MeKong_"
                f"{station.replace(' ', '')}_"
                f"_"
                f"ft{self._configs.features}_"
                f"sl{self._configs.seq_len}_"
                f"ll{self._configs.label_len}_"
                f"pl{self._configs.pred_len}_"
                f"dm{self._configs.d_model}_"
                f"nh{self._configs.n_heads}_"
                f"el{self._configs.e_layers}_"
                f"dl{self._configs.d_layers}_"
                f"df{self._configs.d_ff}_"
                f"fc{self._configs.factor}_"
                f"eb{self._configs.embed}_"
                f"dt{self._configs.distil}_")

    def _get_station_path2(self, station1, station2):
        return (f"{model_name}_"
                f"MeKong_Cross_"
                f"{station1.replace(' ', '')}_"
                f"{station2.replace(' ', '')}_"
                f"ft{self._configs.features}_"
                f"sl{self._configs.seq_len}_"
                f"ll{self._configs.label_len}_"
                f"pl{self._configs.pred_len}_"
                f"dm{self._configs.d_model}_"
                f"nh{self._configs.n_heads}_"
                f"el{self._configs.e_layers}_"
                f"dl{self._configs.d_layers}_"
                f"df{self._configs.d_ff}_"
                f"fc{self._configs.factor}_"
                f"eb{self._configs.embed}_"
                f"dt{self._configs.distil}_")

    def _read_results(self, station_results_path):
        file_path = os.path.join(self._results_path, station_results_path)
        pred_npy = os.path.join(file_path, "pred.npy")
        true_npy = os.path.join(file_path, "true.npy")
        metrics_npy = os.path.join(file_path, "metrics.npy")

        pred = np.load(pred_npy)
        true = np.load(true_npy)
        metrics = np.load(metrics_npy)  # metric = np.array([mae, mse, rmse, mape, mspe]
        return pred, true, metrics

    def _write_results_to_lines(self, table, target, rmse, mse, mae):
        lines = table.split('\n')
        target_line_index = None
        for i, line in enumerate(lines):
            if target + " &" in line:
                target_line_index = i
                break

        if target_line_index is not None:
            columns = lines[target_line_index].split('&')

            columns[1] = f" {rmse:.2f} "
            columns[2] = f" {mse:.2f} "
            columns[3] = f" {mae:.2f} "
            columns[3] += rf"\\ \midrule"

            modified_line = "&".join(columns)
            lines[target_line_index] = modified_line
        else:
            raise ValueError(f"{target} not exist in the table")
        return lines

    def _get_single_station_results(self, station):
        station_results_path = self._get_station_path1(station)
        pred, true, metrics = self._read_results(station_results_path)
        mae = metrics[0]
        mse = metrics[1]
        rmse = metrics[2]
        return pred, true, mae, mse, rmse

    def _get_cross_station_results(self, station):
        if station not in self._station2_list:
            station_results_path = self._get_station_path1(station)
            pred, true, metrics = self._read_results(station_results_path)
        else:
            pred_list = []
            # predict from itself
            station_results_path = self._get_station_path1(station)
            pred, true, metrics = self._read_results(station_results_path)
            pred_list.append(pred)
            # predict from other stations
            for (station1, station2) in self._cross_list:
                if station == station2:
                    station_results_path = self._get_station_path2(station1, station2)
                    pred, true, metrics = self._read_results(station_results_path)
                    pred_list.append(pred)
            avg_pred = np.mean(pred_list, axis=0)
            metrics = metric(avg_pred, true)
        mae = metrics[0]
        mse = metrics[1]
        rmse = metrics[2]
        return pred, true, mae, mse, rmse

    def save_init_to_latex_table(self, file_save_name=None):
        if file_save_name is None:
            file_save_name = f"{self._model}_init_pred_table.txt"
        init_table = copy.copy(self._latex_table_template)
        init_table = init_table.replace(r'\caption{}',
                                        r'\caption{' + f'The initial prediction results from {self._model}.' + '}')
        init_table = init_table.replace(r'\label{}',
                                        r'\label{' + f'{self._model}_init_pred' + '}')

        mae_list = []
        mse_list = []
        rmse_list = []
        for station in self._all_stations:
            pred, true, mae, mse, rmse = self._get_single_station_results(station)
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)

            lines = self._write_results_to_lines(init_table, station, rmse, mse, mae)
            init_table = '\n'.join(lines)

        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        lines = self._write_results_to_lines(init_table, 'Avg', avg_rmse, avg_mse, avg_mae)
        init_table = '\n'.join(lines)

        with open(os.path.join(self._tables_save_path, file_save_name), "w") as text_file:
            text_file.write(init_table)
        return init_table

    def save_cross_to_latex_table(self, file_save_name=None):
        if file_save_name is None:
            file_save_name = f"{self._model}_cross_pred_table.txt"
        cross_table = copy.copy(self._latex_table_template)
        cross_table = cross_table.replace(r'\caption{}',
                                          r'\caption{' + f'The cross prediction results from {self._model}.' + '}')
        cross_table = cross_table.replace(r'\label{}',
                                          r'\label{' + f'{self._model}_cross_pred' + '}')

        mae_list = []
        mse_list = []
        rmse_list = []

        for station in self._all_stations:
            pred, true, mae, mse, rmse = self._get_cross_station_results(station)
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)

            lines = self._write_results_to_lines(cross_table, station, rmse, mse, mae)
            cross_table = '\n'.join(lines)

        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        lines = self._write_results_to_lines(cross_table, 'Avg', avg_rmse, avg_mse, avg_mae)
        cross_table = '\n'.join(lines)

        with open(os.path.join(self._tables_save_path, file_save_name), "w") as text_file:
            text_file.write(cross_table)
        return cross_table

    def save_init_to_csv(self, file_save_name=None):
        if file_save_name is None:
            file_save_name = f"{self._model}_init_pred_table.csv"
        rows = []
        header = ['Station', 'RMSE', 'MSE', 'MAE']
        rows.append(header)

        mae_list = []
        mse_list = []
        rmse_list = []
        for station in self._all_stations:
            pred, true, mae, mse, rmse = self._get_single_station_results(station)
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            rows.append([station, rmse, mse, mae])
        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        rows.append(['Avg', avg_rmse, avg_mse, avg_mae])
        with open(os.path.join(self._csv_save_path, file_save_name), "w", newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    def save_cross_to_csv(self, file_save_name=None):
        if file_save_name is None:
            file_save_name = f"{self._model}_cross_pred_table.csv"
        rows = []
        header = ['Station', 'RMSE', 'MSE', 'MAE']
        rows.append(header)

        mae_list = []
        mse_list = []
        rmse_list = []
        for station in self._all_stations:
            pred, true, mae, mse, rmse = self._get_cross_station_results(station)
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            rows.append([station, rmse, mse, mae])
        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        rows.append(['Avg', avg_rmse, avg_mse, avg_mae])
        with open(os.path.join(self._csv_save_path, file_save_name), "w", newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    def draw_cross_pred(self, station):
        pred, true, mae, mse, rmse = self._get_cross_station_results(station)
        plt.plot(pred[:, 0, 0])
        plt.plot(true[:, 0, 0])
        plt.show()


if __name__ == '__main__':
    results_path = os.path.join('..', 'results', 'PatchTST')
    model_name = 'PatchTST'
    make_table = MakeTable(model_name, results_path)

    make_table.save_init_to_latex_table()
    make_table.save_cross_to_latex_table()
    make_table.save_init_to_csv()
    make_table.save_cross_to_csv()
