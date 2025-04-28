import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class USC_dataset(Dataset): 
    
    def __init__(self, args, root_path, 
                flag='train', 
                size=None,
                features='S', 
                data_path='ETTm1.csv',
                target='OT', 
                scale=True,
                timeenc=0, 
                freq='t', 
                seasonal_patterns=None):
        
        self.args = args
        self.features = args.features
        self.root_path = args.root_path
        self.X_filename = f'seq{args.seq_len}_pred{args.seq_len}_X.npy'
        self.Y_filename = f'seq{args.seq_len}_pred{args.seq_len}_Y.npy'
        self.use_action_progress = args.use_action_progress
        self.use_offense = args.use_offense 
        self.consider_only_offense = args.consider_only_offense 
        
        self.scale = scale
        
        # Dowsample factor for Graph NN 
        self.downsample_factor = 4 if args.model[:4] == 'ST_G' else 1
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
        self.scaler= {'mean' : [], 'scale':[]} 
        self.data_x = np.load(os.path.join(self.root_path, self.X_filename))
        self.data_y = np.load(os.path.join(self.root_path, self.Y_filename))

        # Args correction and offense filter
        if self.consider_only_offense :
            off_mask = self.data_x[:,0,2].astype(bool)
            self.data_x = self.data_x[off_mask]
            self.data_y = self.data_y[off_mask]
            self.use_offense = False
        if self.features == 'MS' :
            self.use_action_progress = True
            
        # Downsampling if necessary    
        downsample_mask = [i % self.downsample_factor == 0 for i in range(self.data_x.shape[0])]
        self.data_x = self.data_x[downsample_mask]
        self.data_y = self.data_y[downsample_mask]

        # TDO : makes this parametrizable
        train_share, val_share, test_share = 0.7, 0.15, 0.15 
        n = self.data_x.shape[0]
        split_indices = [0 , int(np.floor(n * train_share)), int(np.floor(n * (train_share + val_share))), n] 
        # Columns order :
        ### 0 : action id
        ### 1 : target
        ### 2 : offense
        
         
        mask = [False, self.use_action_progress, self.use_offense] + [True] * 60
        target_index = 1 #Action progression is in the second column on the last dimension                      
                              
        if self.features == 'M': # multivariate precicts multivariate
            self.data_x = self.data_x[:,:,mask]
            self.data_y = self.data_y[:,:,mask]
        elif self.features == 'MS' : # multivariate precicts univariate
            self.data_x = self.data_x[:,:,mask]
            self.data_y = np.expand_dims(self.data_y[:,:,target_index], 2)      
        elif self.features == 'S': # univariate precicts univariate
            self.data_x = np.expand_dims(self.data_x[:,:,target_index], 2)
            self.data_y = np.expand_dims(self.data_y[:,:,target_index], 2)

        if self.scale:
            if not self.features == 'MS' :
                train_data = np.concatenate((self.data_x[split_indices[0] : split_indices[1]],
                                         self.data_y[split_indices[0] : split_indices[1]]), axis = 1)
            else :
                train_data = self.data_x[split_indices[0] : split_indices[1]]
            train_data = train_data.reshape((-1,train_data.shape[2]))
            start_col = 0
            if self.use_action_progress or self.features == 'S' :
                self.scaler['mean'].append(train_data[:,0].mean())
                self.scaler['scale'].append(train_data[:,0].std())
                # No scaling of the target feature as it is a difference between the end of sequence and value at all points.
                #self.scaler['mean'].append(0)
                #self.scaler['scale'].append(1)                
                start_col +=1
            if self.use_offense :
                self.scaler['mean'].append(0)
                self.scaler['scale'].append(1)
                start_col +=1
            if self.features == 'M' or self.features == 'MS' :            
                for i in range(15):
                    # x scaling
                    self.scaler['mean'].append(50) # Center on the x-axis
                    self.scaler['scale'].append(50)
                    # y scaling
                    self.scaler['mean'].append(35) # Center on the y-axis
                    self.scaler['scale'].append(50) # Same scale as for x not to artificially increase the y displacement                        
                    # v scaling
                    self.scaler['mean'].append(train_data[:,start_col + 4*i + 2].mean())
                    self.scaler['scale'].append(train_data[:,start_col + 4*i + 2].std())
                    # goal_angle_scaling
                    self.scaler['mean'].append(0)
                    self.scaler['scale'].append(180) # Angles ranges from -180 to 180
            
            #offset_x = 1 if self.use_action_progress and (self.features == 'M' or self.feature == 'MS') else 0
            #offset_y = 1 if self.use_action_progress  and self.features == 'M' else 0
            #print(offset_x, offset_y)
            for i in range(self.data_x.shape[2]):
                self.data_x[:,:,i] = (self.data_x[:,:,i] - self.scaler['mean'][i]) / self.scaler['scale'][i]
            for i in range(self.data_y.shape[2]):
                self.data_y[:,:,i] = (self.data_y[:,:,i] - self.scaler['mean'][i]) / self.scaler['scale'][i]
            
            self.scaler['offset_x'] = 0
            self.scaler['offset_y'] = 0
            self.scaler['x_size'] = self.args.batch_size * self.data_x.shape[1] * self.data_x.shape[2]
            self.scaler['y_size'] = self.args.batch_size * self.data_y.shape[1] * self.data_y.shape[2]
        
        self.data_x = self.data_x[split_indices[self.set_type] : split_indices[self.set_type +1]]
        self.data_y = self.data_y[split_indices[self.set_type] : split_indices[self.set_type +1]]


    def __getitem__(self, index):
        seq_x = self.data_x[index]
        if self.features == 'MS':
            seq_y =  np.concatenate((np.expand_dims(seq_x[-self.args.label_len:,0],1), self.data_y[index]), axis = 0)
        else :    
            seq_y =  np.concatenate((seq_x[-self.args.label_len:,:], self.data_y[index]), axis = 0)
        # x_mark and y_mark must be of shape [l, 4] and [p+t, 4]
        seq_mark = np.arange(self.args.seq_len + self.args.pred_len) / (self.args.seq_len + self.args.pred_len -1) -0.5
        seq_x_mark = np.concatenate((np.zeros((self.args.seq_len,3)),seq_mark[:self.args.seq_len].reshape(-1,1)), axis = 1)
        seq_y_mark = np.concatenate((np.zeros((self.args.pred_len+self.args.label_len,3)),
                                     seq_mark[self.args.seq_len- self.args.label_len:].reshape(-1,1)), axis = 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark 

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        
        data_size = data.shape[0] * data.shape[1] 
        if not hasattr(self,'inverse_offset') :
            if data_size == self.scaler['y_size'] :
                offset = self.scaler['offset_y']
            elif data_size == self.scaler['x_size'] :
                offset = self.scaler['offset_x']
            else : 
                raise(ValueError('Please investigate scaling operation, sizes do not match with neither x nor y'))
            self.inverse_offset = offset
            
        for i in range(data.shape[1]):
            data[:,i] = data[:,i] * self.scaler['scale'][i+self.inverse_offset] + self.scaler['mean'][i+self.inverse_offset]
        return data   
    

class BBall_dataset(Dataset):
    def __init__(self, args, root_path='.dataset/bball', flag='train', size=None,
                 features='M', data_path='all_data.npy',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = args.seq_len
            self.label_len = args.label_len
            self.pred_len = args.pred_len
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data = np.load(os.path.join(self.root_path,
                                          self.data_path))

        train_share, val_share, test_share = 0.6, 0.2, 0.2 
        n = data.shape[0]
        split_indices = [0 , int(np.floor(n * train_share)), int(np.floor(n * (train_share + val_share))), n] 

        if self.scale:
            # Revoir : problème de dimension
            train_data = data[split_indices[0]:split_indices[1]]
            self.scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
            data = self.scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        else:
            data = data

        # TDO : makes this parametrizable
        self.data_x = data[:,:self.seq_len, :]
        self.data_y = data[:,self.seq_len:self.seq_len+self.pred_len, :]


        if self.features == 'MS':
            self.data_y = self.data_y[:,:,0]
        elif self.features == 'S':
            self.data_x = self.data_x[:,:,0]
            self.data_y = self.data_y[:,:,0]
            
        self.data_x = self.data_x[split_indices[self.set_type] : split_indices[self.set_type +1]]
        self.data_y = self.data_y[split_indices[self.set_type] : split_indices[self.set_type +1]]


    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y =  np.concatenate((seq_x[-self.args.label_len:,:], self.data_y[index]), axis = 0)
        # x_mark and y_mark must be of shape [l, 4] and [p+t, 4]
        seq_mark = np.arange(self.args.seq_len + self.args.pred_len) / (self.args.seq_len + self.args.pred_len -1) -0.5
        seq_x_mark = np.concatenate((np.zeros((self.args.seq_len,3)),seq_mark[:self.args.seq_len].reshape(-1,1)), axis = 1)
        seq_y_mark = np.concatenate((np.zeros((self.args.pred_len+self.args.label_len,3)),
                                     seq_mark[self.args.seq_len- self.args.label_len:].reshape(-1,1)), axis = 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark 

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)