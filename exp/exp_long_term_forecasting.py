from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from utils.log import log
import math

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd)         
        else :
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                if self.args.loss == 'FDE' :
                    loss = criterion(pred[:,-1,:], true[:,-1,:])
                else :
                    loss = criterion(pred, true)
                #print(f"{pred.shape=}, {true.shape=}, {loss=}")

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        time_start = time.time()

        left_time_str = 'not_calculated'
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # ADDED : Scheduler
        if self.args.lr_scheduler :        
            scheduler = torch.optim.lr_scheduler.OneCycleLR(model_optim, 
                                                            max_lr=self.args.learning_rate,
                                                            steps_per_epoch=len(train_loader), 
                                                            epochs=self.args.train_epochs)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if self.args.loss == 'FDE' :
                            loss = criterion(outputs[:,-1,:], batch_y[:,-1,:])
                        else :
                            loss = criterion(outputs, batch_y)                        
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)                    
                    if self.args.loss == 'FDE' :
                        loss = criterion(outputs[:,-1,:], batch_y[:,-1,:])
                    else :
                        loss = criterion(outputs, batch_y)                        
                    train_loss.append(loss.item())


                if (i+1) % (train_steps//10) == 0:
                    #print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    #print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    left_time_str = f"{int(left_time//60)} min {int(left_time%60)} secs" if left_time > 60 else f"{left_time:.1f} secs"
                # print(f"Running epoch {epoch+1} - batch n° {i+1}/{train_steps} - estimated remaining time : {left_time_str}           ", end = '\r')

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                # ADDED : scheduler   
                if self.args.lr_scheduler :
                    scheduler.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            es_message = early_stopping(vali_loss, self.model, path, args = self.args, epoch=epoch)
            epoch_time = time.time() - epoch_time
            duration_str = f"{int(epoch_time//60)} min {int(epoch_time%60)} secs" if epoch_time > 60 else f"{epoch_time:.1f} secs"

            #print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #print("Epoch: {0}, Future Learning Rate : {1:.4e},Steps: {2} | Train Loss: {3:.4e} Vali Loss: {4:.4e} Test Loss: {5:.4e}".format(
            #    epoch + 1, model_optim.param_groups[0]['lr'],train_steps, train_loss, vali_loss, test_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if  math.isnan(vali_loss):
                print("Model diverges, validation loss is nan")
                break

            # ADDED : scheduler
            if not self.args.lr_scheduler :
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            print("Epoch: {0} | duration: {1} | train loss: {2:.2e} - val loss: {3:.2e} | {4} | next Learning Rate: {5:.2e}".format(epoch + 1, 
                                                                                                                                            duration_str,
                                                                                                                                            train_loss,
                                                                                                                                            vali_loss,
                                                                                                                                            #test_loss,
                                                                                                                                            es_message,
                                                                                                                                            model_optim.param_groups[0]['lr']
                                                                                                                                            ),
                  end = '\n')


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.args.fit_time = np.round(time.time() - time_start,2) 

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = self.args.results_path + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs_shape = outputs.shape
                    batch_y_shape = batch_y.shape
                    outputs = test_data.inverse_transform(outputs.reshape(outputs_shape[0] * outputs_shape[1], -1)).reshape(outputs_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(batch_y_shape[0] * batch_y_shape[1], -1)).reshape(batch_y_shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        #print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('test shape:', preds.shape, trues.shape)

        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            

        mae, mse, rmse, mape, mspe, fde = metric(preds, trues)
        print('fde:{:.4e}, rmse:{:.4e}, mse:{:.4e}, mae:{:.4e}, dtw:{}'.format(fde,rmse,mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        # logging
        trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_param_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        non_trainable_param_count += sum(b.numel() for b in self.model.buffers())
        metrics = {'nb_params':trainable_param_count+ non_trainable_param_count, 
                   'nb_tr_params':trainable_param_count,
                   'nb_nontr_params':non_trainable_param_count,
                   'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe, 'fde':fde}
        
        log(metrics=metrics, args = self.args)

        return
