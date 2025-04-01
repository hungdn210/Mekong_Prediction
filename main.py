from utils.configs import BasicConfigs
from utils.constants import Constants
from exp.exp_mekong import Exp_MeKong
from exp.exp_mekong_phase_a import Exp_MeKong as Exp_MeKong_phase_a
import time


class MeKongWaterLevelPrediction:
    def __init__(self):
        self.args = BasicConfigs()
        self.constants = Constants()

        # all single station list
        self._all_stations = self.constants.all_stations
        # cross station
        self._cross_list = self.constants.cross_list

        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ', '')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]

        if self.args.use_gpu:
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            print('Use CPU')

    def _print_properties(self, obj):
        cls = type(obj)

        properties = {}
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, property):
                try:
                    value = getattr(obj, name)
                    properties[name] = value
                except Exception as e:
                    properties[name] = f"<Error: {str(e)}>"

        for name, value in properties.items():
            print(f"  - {name}: {value}")

    def station_run(self, station1, station2=None, verbose=True):
        if station2 is None:
            station2 = ''
        # data setting
        data = 'MeKong'
        data1_path = f'{station1}.csv'
        data2_path = ''
        if not station2 == '':
            data = 'MeKong_Cross'
            data2_path = f'{station2}.csv'
            if verbose:
                print(f"Running cross station {station1} and {station2}")
        else:
            if verbose:
                print(f"Running single station {station1}")
        self.args.data = data
        self.args.data1_path = data1_path
        self.args.data2_path = data2_path
        # exp setting
        exp = Exp_MeKong_phase_a(self.args, verbose)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            self.args.model,
            self.args.data,
            station1.replace(' ', ''),
            station2.replace(' ', ''),
            self.args.features,
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
        if verbose:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(setting))
        exp.train(setting)
        if verbose:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n'.format(setting))
        exp.test(setting)

    def run_all_stations(self):
        self.args.model_default_configs(self.args.model)
        self._print_properties(self.args)

        all_tasks = []
        for station1 in self._all_stations:
            station2 = None
            all_tasks.append((station1, station2))

        print(f"Start to run {len(self._all_stations)} stations")
        print(f"Overall {len(all_tasks)} tasks")
        start_time = time.time()
        task_time = None
        all_duration = 0
        completed_count = 0
        for station1, station2 in all_tasks:
            # run each task
            self.station_run(station1, station2, verbose=False)
            # count completed tasks, calc time
            completed_count += 1
            if task_time is None:
                task_time = start_time
            duration = time.time() - task_time
            all_duration = time.time() - start_time
            print(f"Progress: {completed_count}/{len(all_tasks)}, "
                  f"task duration: {duration:.2f}s, "
                  f"all duration: {all_duration:.2f}s\n")
            task_time = time.time()

        print(f"All tasks finished! All time consumed: {all_duration:.2f}s")


if __name__ == '__main__':
    forecast = MeKongWaterLevelPrediction()
    forecast.station_run('Chiang Saen')
