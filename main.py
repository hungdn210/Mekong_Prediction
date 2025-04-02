from utils.constants import Constants
from utils.print_args import print_args
from exp.exp_mekong import Exp_MeKong as Exp_MeKong_initial
from exp.exp_mekong_phase_a import Exp_MeKong as Exp_MeKong_phase_a
import time
import torch
import os
import argparse
import yaml
import random
import numpy as np


def parse_args():
    # base configs（from YAML file）
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('-c', '--config', type=str, default='configs/base_configs.yaml')
    base_args, _ = base_parser.parse_known_args()

    # load YAML configs
    with open(base_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # full configs（combine YAML and CMD configs
    parser = argparse.ArgumentParser(description='Water Level Prediction')
    # task setting
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='Linear')
    parser.add_argument('--station', type=str, default='all')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--run_initial', action='store_true', help='initial', default=False)
    parser.add_argument('--run_phase_a', action='store_true', help='phase a', default=False)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--des', type=str, default='exp')
    # model params
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--factor', type=int, default=1)

    parser.set_defaults(**yaml_config)  # load YAML as default
    return parser.parse_args()


class MeKongWaterLevelPrediction:
    def __init__(self, args):
        self.args = args
        # all stations list
        self._all_stations = Constants().all_stations
        self._verbose = args.verbose

    def _run_single_station(self, station, exp):
        if station not in self._all_stations:
            raise ValueError(f"Station {station} not found.")

        self.args.data_path = station + '.csv'
        print(f"Running {station} station")
        # exp setting
        setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            self.args.model,
            station.replace(' ', ''),
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des
        )
        if self.args.is_training:
            if self._verbose:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(setting))
            exp.train(setting)
            if self._verbose:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n'.format(setting))
            mse, mae = exp.test(setting)
            torch.cuda.empty_cache()
        else:
            if self._verbose:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n'.format(setting))
            mse, mae = exp.test(setting)
            torch.cuda.empty_cache()
        return mse, mae

    def _run_tasks(self, phase, exp):
        if self.args.station == 'all':
            print(f"Running all stations in {phase}")
            files = os.listdir(self.args.root_path)
            station_list = [file.split('.csv')[0] for file in files if file.endswith('.csv')]
            start_time = time.time()
            for i, station in enumerate(station_list):
                task_start_time = time.time()
                mse, mae = self._run_single_station(station, exp)
                duration = time.time() - task_start_time
                overall_duration = time.time() - start_time
                print(f"tasks: {i + 1} / {len(station_list)}, "
                      f"duration: {duration:.2f}s, "
                      f"overall time: {overall_duration:.2f}s, "
                      f"mse: {mse:.4f}, mae: {mae:.4f}")
            print(f"Run all stations in {phase} done")
        else:
            print(f"Running station {self.args.station} in {phase}")
            task_start_time = time.time()
            mse, mae = self._run_single_station(self.args.station, exp)
            duration = time.time() - task_start_time
            print(f"duration: {duration:.2f}s, "
                  f"mse: {mse:.4f}, mae: {mae:.4f}")
            print(f"Run station {self.args.station} in {phase} done")

    def run(self):
        if self.args.run_initial:
            args.data = 'Initial'
            exp = Exp_MeKong_initial(self.args, self._verbose)
            self._run_tasks('Initial', exp)
        if self.args.run_phase_a:
            args.data = 'PhaseA'
            exp = Exp_MeKong_phase_a(self.args, self._verbose)
            self._run_tasks('Phase A', exp)


if __name__ == '__main__':
    args = parse_args()
    args.use_gpu = torch.cuda.is_available()
    args.gpu = torch.cuda.current_device() if args.use_gpu else -1
    print(torch.cuda.is_available())  # should be True
    print(torch.cuda.get_device_name(0)) 
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    print_args(args)

    forcast = MeKongWaterLevelPrediction(args)
    forcast.run()
