import os
import numpy as np
from utils.constants import Constants
import copy
import csv


class MakeTable:
    def __init__(self, model, results_path, setting, tables_save_path=None, csv_save_path=None):
        if tables_save_path is None:
            tables_save_path = os.path.join('latex_tables')
        if csv_save_path is None:
            csv_save_path = os.path.join('csv_files')

        self._model = model
        self._results_path = results_path
        self._setting = setting
        self._tables_save_path = tables_save_path
        self._csv_save_path = csv_save_path

        if not os.path.exists(self._tables_save_path):
            os.makedirs(self._tables_save_path)
        if not os.path.exists(self._csv_save_path):
            os.makedirs(self._csv_save_path)
        self._all_stations = Constants().all_stations

        self._latex_table_template = self._create_table_template()

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

    def _get_station_path(self, station):
        a = self._setting.split('_sl')[0]
        b = self._setting.split('_sl')[1]
        return a + station.replace(' ', '') + '_sl' + b

    def _read_results(self, phase, station_results_path):
        file_path = os.path.join(self._results_path, station_results_path, phase)
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

    def _get_single_station_results(self, phase, station):
        station_results_path = self._get_station_path(station)
        pred, true, metrics = self._read_results(phase, station_results_path)
        mae = metrics[0]
        mse = metrics[1]
        rmse = metrics[2]
        return pred, true, mae, mse, rmse

    def save_to_latex_table(self, phase='initial', file_save_name=None):
        if file_save_name is None:
            file_save_name = f"{self._model}_{phase}.txt"
        init_table = copy.copy(self._latex_table_template)
        init_table = init_table.replace(r'\caption{}',
                                        r'\caption{' + f'The {phase} prediction results from {self._model}.' + '}')
        init_table = init_table.replace(r'\label{}',
                                        r'\label{' + f'{self._model}_init_pred' + '}')

        mae_list = []
        mse_list = []
        rmse_list = []
        for station in self._all_stations:
            pred, true, mae, mse, rmse = self._get_single_station_results(phase, station)
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

    def save_to_csv(self, phase='initial', file_save_name=None):
        if file_save_name is None:
            file_save_name = f"{self._model}_{phase}.csv"
        rows = []
        header = ['Station', 'RMSE', 'MSE', 'MAE']
        rows.append(header)

        mae_list = []
        mse_list = []
        rmse_list = []
        for station in self._all_stations:
            pred, true, mae, mse, rmse = self._get_single_station_results(phase, station)
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


if __name__ == '__main__':
    results_path = os.path.join('..', 'results')
    model_name = 'RLinear'
    setting = 'RLinear__sl96_ll0_pl12_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_exp'
    make_table = MakeTable(model_name, results_path, setting)

    make_table.save_to_latex_table(phase='initial')
    make_table.save_to_csv(phase='initial')

    make_table.save_to_latex_table(phase='phase_a')
    make_table.save_to_csv(phase='phase_a')
