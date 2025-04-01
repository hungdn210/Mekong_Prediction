import gc

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping_mekong_phase_a, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.WaterLevelChannelFusion import Model as WaterLevelChannelFusion

warnings.filterwarnings('ignore')


class Exp_MeKong(Exp_Basic):
    def __init__(self, args, verbose=True):
        super(Exp_MeKong, self).__init__(args, verbose=verbose)
        self.device = self._acquire_device()
        self.models_dic = self._build_models_dict()

    def _build_models_dict(self):
        models_dic = {}
        models_dic['channel_fusion'] = WaterLevelChannelFusion().float()
        for name in ['water_level', 'water_discharge', 'rainfall']:
            models_dic[name] = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            for name in ['water_level', 'water_discharge', 'rainfall']:
                models_dic[name] = nn.DataParallel(models_dic[name], device_ids=self.args.device_ids)
        for k in models_dic.keys():
            models_dic[k] = models_dic[k].to(self.device)
        return models_dic

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.verbose)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, criterion):
        total_loss = []
        for name in self.models_dic.keys():
            self.models_dic[name].eval()
        with torch.no_grad():
            for i, (wl_x, wd_x, rf_x, wl_y) in enumerate(self.vali_loader):
                wl_x = wl_x.float().to(self.device)
                wl_y = wl_y.float().to(self.device)
                # encoder - decoder
                wl_outputs = self.models_dic['water_level'](wl_x)
                outputs = wl_outputs
                if wd_x is not None:
                    wd_x = wd_x.float().to(self.device)
                    wd_outputs = self.models_dic['water_discharge'](wd_x)
                    outputs = torch.cat([outputs, wd_outputs], dim=-1)
                if rf_x is not None:
                    rf_x = rf_x.float().to(self.device)
                    rf_outputs = self.models_dic['water_discharge'](rf_x)
                    outputs = torch.cat([outputs, rf_outputs], dim=-1)
                # channel fusion
                fusion_pred = self.models_dic['channel_fusion'](outputs)

                pred = fusion_pred.detach().cpu()
                true = wl_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        for name in self.models_dic.keys():
            self.models_dic[name].train()
        return total_loss

    def train(self, setting):
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping_mekong_phase_a(patience=self.args.patience, verbose=self.verbose, delta=1e-6)

        model_optim_dic = {}
        for name, model in self.models_dic.items():
            model_optim_dic[name] = self._select_optimizer(model)
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for i, (wl_x, wd_x, rf_x, wl_y) in enumerate(self.train_loader):
                outputs = None  # concat 3 channels' output
                if wl_x is not None:
                    self.models_dic['water_level'].train()
                    model_optim_dic['water_level'].zero_grad()
                    wl_x, wl_y = wl_x.float().to(self.device), wl_y.float().to(self.device)
                    wl_outputs = self.models_dic['water_level'](wl_x)
                    loss = criterion(wl_outputs, wl_y)
                    train_loss.append(loss.item())
                    loss.backward(retain_graph=True)
                    model_optim_dic['water_level'].step()
                    if outputs is None:
                        outputs = wl_outputs
                    else:
                        outputs = torch.cat([outputs, wl_outputs], dim=-1)
                if wd_x is not None:
                    self.models_dic['water_discharge'].train()
                    model_optim_dic['water_discharge'].zero_grad()
                    wd_x, wl_y = wd_x.float().to(self.device), wl_y.float().to(self.device)
                    wd_outputs = self.models_dic['water_discharge'](wd_x)
                    loss = criterion(wd_outputs, wl_y)
                    train_loss.append(loss.item())
                    loss.backward(retain_graph=True)
                    model_optim_dic['water_discharge'].step()
                    if outputs is None:
                        outputs = wd_outputs
                    else:
                        outputs = torch.cat([outputs, wd_outputs], dim=-1)
                if rf_x is not None:
                    self.models_dic['rainfall'].train()
                    model_optim_dic['rainfall'].zero_grad()
                    rf_x, wl_y = rf_x.float().to(self.device), wl_y.float().to(self.device)
                    rf_outputs = self.models_dic['rainfall'](rf_x)
                    loss = criterion(rf_outputs, wl_y)
                    train_loss.append(loss.item())
                    loss.backward(retain_graph=True)
                    model_optim_dic['rainfall'].step()
                    if outputs is None:
                        outputs = rf_outputs
                    else:
                        outputs = torch.cat([outputs, rf_outputs], dim=-1)
                # if 3 channels are all none, skip this epoch
                if outputs is None:
                    continue

                iter_count += 1
                # channel fusion
                self.models_dic['channel_fusion'].train()
                model_optim_dic['channel_fusion'].zero_grad()
                preds = self.models_dic['channel_fusion'](outputs)
                loss = criterion(preds, wl_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim_dic['channel_fusion'].step()

                if self.verbose:
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            if self.verbose:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(criterion)
            test_loss = self.vali(criterion)
            if self.verbose:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.models_dic, path)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

            for name in self.models_dic.keys():
                adjust_learning_rate(model_optim_dic[name], epoch + 1, self.args, self.verbose)

        for name in self.models_dic.keys():
            ckpt_name = f'checkpoint_{name}.pth'
            best_model_path = path + '/phase_a/' + ckpt_name
            self.models_dic[name].load_state_dict(torch.load(best_model_path))

    def test(self, setting, test=0):
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')

        if self.verbose:
            print('loading model')
        self.models_dic = self._build_models_dict()
        for name in self.models_dic.keys():
            ckpt_name = f'checkpoint_{name}.pth'
            self.models_dic[name].load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'phase_a', ckpt_name)
            ))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/phase_a/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for name in self.models_dic.keys():
            self.models_dic[name].eval()
        with torch.no_grad():
            for i, (wl_x, wd_x, rf_x, wl_y) in enumerate(self.test_loader):
                wl_x = wl_x.float().to(self.device)
                wl_y = wl_y.float().to(self.device)
                # encoder - decoder
                wl_outputs = self.models_dic['water_level'](wl_x)
                outputs = wl_outputs
                if wd_x is not None:
                    wd_x = wd_x.float().to(self.device)
                    wd_outputs = self.models_dic['water_discharge'](wd_x)
                    outputs = torch.cat([outputs, wd_outputs], dim=-1)
                if rf_x is not None:
                    rf_x = rf_x.float().to(self.device)
                    rf_outputs = self.models_dic['water_discharge'](rf_x)
                    outputs = torch.cat([outputs, rf_outputs], dim=-1)
                # channel fusion
                fusion_pred = self.models_dic['channel_fusion'](outputs)
                fusion_pred = fusion_pred.detach().cpu().numpy()
                wl_y = wl_y.detach().cpu().numpy()
                if self.test_data.scale and self.args.inverse:
                    shape = fusion_pred.shape
                    fusion_pred = self.test_data.inverse_transform(
                        fusion_pred.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    wl_y = self.test_data.inverse_transform(wl_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                pred = fusion_pred
                true = wl_y
                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                input = wl_x.detach().cpu().numpy()
                if self.test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = self.test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if self.verbose:
            print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if self.verbose:
            print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/phase_a/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        if self.verbose:
            print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_mekong_phase_a.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
