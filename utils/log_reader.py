import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
from data_provider.data_factory import data_provider

LOG_PATH = './logs'
DATA_PATH = "./dataset/USC"
LOCAL_START_INDEX = 232
BASIS_ARGS_LOGS = 207
CALMIP_START_INDEX = {
    2: 471,
    3 : 1392,
    4 : 1604,
    5 : 2304,
    6 : np.inf
    }


class BaselineScorer:
    
    def __init__(self, time_horizons, metric:str = 'fde'):
        self.ths = time_horizons
        self.metric = metric
        self.args = SimpleNamespace(**{"root_path" : DATA_PATH, 
                                       'data_path' : 'na',
                                       "data":'USC', 
                                       'seq_len':0, 
                                       'pred_len':0, 
                                       'label_len':0,
                                       'scale':False,
                                       # Default args to allow loader to work
                                       'model':'Transformer',
                                       'task_name':'long_term_forecast',
                                       'embed':'timeF',
                                       'features':'S',
                                       'use_action_progress':True,
                                       'use_offense':False,
                                       'consider_only_offense':True,
                                       'batch_size':256,
                                       'freq':'h',
                                       'seasonal_patterns':'Monthly',
                                       'num_workers' : 1,
                                       'target':'na'})
        self.score_baselines()
        
    @staticmethod
    def score_baseline(args, seq_len : int, pred_len : int):
        args.seq_len = seq_len
        args.pred_len = pred_len
        uscds, _ = data_provider(args= args, flag = 'test', )
        ### StandStill
        ss_fde = np.sqrt(np.mean(uscds.data_y[:,-1,0]**2))
        ss_rmse = np.sqrt(np.mean(uscds.data_y[:,:,0]**2))
        ### Constant Velocity
        bl_cv_delta = (uscds.data_x[:,-1,0] - uscds.data_x[:,0,0]) / uscds.data_x.shape[1]
        bl_cv_pred = np.repeat(uscds.data_x[:,-1,0].reshape(-1,1,1), uscds.data_y.shape[1], axis=1)
        for i in range(uscds.data_y.shape[1]):
            bl_cv_pred[:,i,0] = (i+1) * bl_cv_delta 
        cv_fde = np.sqrt(np.mean( (uscds.data_y[:,-1,0] - bl_cv_pred[:,-1,0])**2 ))
        cv_rmse = np.sqrt(np.mean( (uscds.data_y[:,:,0] - bl_cv_pred[:,:,0])**2 ))
        ### Concat
        bls = [{'model':'Baseline - StandStill', 'features':'S', 'time':f"{seq_len}-{pred_len}",'fde':ss_fde, 'rmse':ss_rmse},
            {'model':'Baseline - ConstantVelocity', 'features':'S', 'time':f"{seq_len}-{pred_len}",'fde':cv_fde, 'rmse': cv_rmse}]
        
        return bls
      
    def score_baselines(self):
        bls = []
        for th in self.ths :
            bls += self.score_baseline(self.args, th,th)
        self.bls = pd.DataFrame(bls)
        self.bls = self.bls[['model', 'features', 'time', self.metric]].rename(columns = {self.metric : 'mean'})
        # self.bls = self.bls.pivot(index = ['model','features'], columns = 'time', values= 'mean')  
        # ordered_cols = [f"{th}-{th}" for th in self.ths]
        # self. bls = self.bls[ordered_cols]

class ResultReader:
    
    def __init__(self, 
                 time_horizons:list,
                 runs : list,
                 metric : str = 'fde',
                 target_n_exp:int = 3):
        self.df = pd.read_csv(os.path.join(LOG_PATH,'calmip_logs.csv'), index_col = 0)
        self.runs = runs
        self.metric = metric
        self.ths = time_horizons
        self._assign_logs_to_run()
        self._query_runs()
        self._add_model_identifier()
        self._filter_nb_exp()
        self.df['time'] = self.df.apply(lambda row : f"{row['seq_len']}-{row['pred_len']}", axis =1)
        self.score_baselines()
        self.read_logs()
    
    # Logs processing
            
    def _assign_logs_to_run(self):
        self.df['run'] = 0
        for index, row in self.df.iterrows():
            if index < CALMIP_START_INDEX[2]:
                self.df.loc[index,'run'] = 1
            elif index < CALMIP_START_INDEX[3]:
                self.df.loc[index,'run'] = 2
            elif index < CALMIP_START_INDEX[5] :
                if row['features'] == 'MS' and row['dec_in'] == 1 :
                    self.df.loc[index,'run'] = 3
                else :
                    self.df.loc[index,'run'] = 4        
            else  :
                self.df.loc[index,'run'] = 5

    def _query_runs(self):
        if not 'run' in self.df.columns:
            self._assign_logs_to_run()
        runs_query = (self.df['run'].isin(self.runs))
        self.df = self.df[runs_query]

    @staticmethod
    def create_model_identifier(row):
        return f"run{row['run']}_{row['model']}_{row['features']}_{row['seq_len']}-{row['pred_len']}_el{row['e_layers']}_dm{row['d_model']}_nh{row['n_heads']}_dl{row['d_layers']}_ff{row['d_ff']}_do{row['dropout']}"

    @staticmethod
    def create_model_architecture(row):
        return f"el{row['e_layers']}_dm{row['d_model']}_nh{row['n_heads']}_dl{row['d_layers']}_ff{row['d_ff']}"

    def _add_model_identifier(self):
        self.df['model_id'] = self.df.apply(lambda row : self.create_model_identifier(row), axis =1)
        self.df['arch'] = self.df.apply(lambda row : self.create_model_architecture(row), axis =1)
        
    def _filter_nb_exp(self):
        exp_df = self.df.copy()
        exp_df = exp_df[exp_df[self.metric].isna()==False]
        exp_df = exp_df.groupby('model_id').count()[['model']]
        exp_df['nb_exp_filter'] = exp_df['model'] >= 3
        self.df = self.df.merge(exp_df['nb_exp_filter'], left_on = 'model_id', right_index = True)
        
    # Baselines
    @staticmethod
    def score_baseline(args, seq_len : int, pred_len : int):
        args.seq_len = seq_len
        args.pred_len = pred_len
        args.root_path = DATA_PATH
        args.scale = False
        uscds, uscdl = data_provider(args= args, flag = 'test', )
        ### StandStill
        ss_fde = np.sqrt(np.mean(uscds.data_y[:,-1,0]**2))
        ss_rmse = np.sqrt(np.mean(uscds.data_y[:,:,0]**2))
        ### Constant Velocity
        bl_cv_delta = uscds.data_x[:,-1,0] - uscds.data_x[:,-2,0]
        bl_cv_pred = np.repeat(uscds.data_x[:,-1,0].reshape(-1,1,1), uscds.data_y.shape[1], axis=1)
        for i in range(uscds.data_y.shape[1]):
            bl_cv_pred[:,i,0] = (i+1) * bl_cv_delta 
        cv_fde = np.sqrt(np.mean( (uscds.data_y[:,-1,0] - bl_cv_pred[:,-1,0])**2 ))
        cv_rmse = np.sqrt(np.mean( (uscds.data_y[:,:,0] - bl_cv_pred[:,:,0])**2 ))
        ### Concat
        bls = [{'model':'Baseline - StandStill', 'features':'S', 'time':f"{seq_len}-{pred_len}",'fde':ss_fde, 'rmse':ss_rmse},
            {'model':'Baseline - ConstantVelocity', 'features':'S', 'time':f"{seq_len}-{pred_len}",'fde':cv_fde, 'rmse': cv_rmse}]
        
        return bls
      
    def score_baselines(self):
        args = SimpleNamespace(**self.df.iloc[0].to_dict())
        bls = []
        for th in self.ths :
            bls += self.score_baseline(args, th,th)
        self.bls = pd.DataFrame(bls)
        self.bls = self.bls[['model', 'features', 'time', self.metric]].rename(columns = {self.metric : 'mean'})
        self.bls['std'] = 0
        self.bls['run'] = 'n.a.'
        self.bls['arch'] = 'n.a.'
        self.bls = self.bls.pivot(index = ['model','features'], columns = 'time', values= ['mean', 'std']).swaplevel(0,1,axis = 1).sort_index(axis = 1)
 
    # Log processing and concatenation with baselines
    def read_logs(self):
        temp_df = self.df[self.df['nb_exp_filter']][['model_id','run','arch','model','features','time',self.metric]].groupby('model_id').agg(
            model=pd.NamedAgg(column='model', aggfunc='first'),
            run=pd.NamedAgg(column='run', aggfunc='first'),
            arch=pd.NamedAgg(column='arch', aggfunc='first'),
            features=pd.NamedAgg(column='features', aggfunc='first'),
            time=pd.NamedAgg(column='time', aggfunc='first'),
            count=pd.NamedAgg(column= self.metric, aggfunc='count'),
            mean=pd.NamedAgg(column= self.metric, aggfunc='mean'),
            std=pd.NamedAgg(column= self.metric, aggfunc='std'),
        ).sort_values(by = 'mean').reset_index()
        temp_df['dupl'] = temp_df[['model','time','features']].duplicated()
        # temp_df['std'] = temp_df['std'].fillna(0)
        summary_df = temp_df[temp_df['dupl'] == False].copy()
        self.results = summary_df.copy()
        summary_df = summary_df.pivot(index = ['model','features'], columns = 'time', values = ['mean','std']).swaplevel(0,1,axis = 1).sort_index(axis = 1)
        summary_df = pd.concat([summary_df,self.bls], axis = 0)            
        summary_df[('Average','mean')] = summary_df.xs('mean', axis = 1, level = 1).mean(axis = 1)
        self.results_df = summary_df.sort_values(by = ('Average','mean'))
           
    def display_logs(self, runs = None):
        if type(runs) == int:
            runs = [runs]
        if runs == None or runs == self.runs :
            return self.results_df
        else :
            display_query = (self.df['run'].isin(runs)) & (self.df['nb_exp_filter'])
            temp_df = self.df[display_query][['model_id','run','arch','model','features','time',self.metric]].groupby('model_id').agg(
                model=pd.NamedAgg(column='model', aggfunc='first'),
                run=pd.NamedAgg(column='run', aggfunc='first'),
                arch=pd.NamedAgg(column='arch', aggfunc='first'),
                features=pd.NamedAgg(column='features', aggfunc='first'),
                time=pd.NamedAgg(column='time', aggfunc='first'),
                count=pd.NamedAgg(column= self.metric, aggfunc='count'),
                mean=pd.NamedAgg(column= self.metric, aggfunc='mean'),
                std=pd.NamedAgg(column= self.metric, aggfunc='std'),
            ).sort_values(by = 'mean').reset_index()
            temp_df['dupl'] = temp_df[['model','time','features']].duplicated()
            # temp_df['std'] = temp_df['std'].fillna(0)
            summary_df = temp_df[temp_df['dupl'] == False].copy()
            summary_df = summary_df.pivot(index = ['model','features'], columns = 'time', values = ['mean','std']).swaplevel(0,1,axis = 1).sort_index(axis = 1)
            summary_df[('Average','mean')] = summary_df.xs('mean', axis = 1, level = 1).mean(axis = 1)
            return summary_df.sort_values(by = ('Average','mean'))


# LaTeX display

def highlight_values(df, n=3, ascending:bool=True, col_list:list=None, n_digits:int = 2):

    # Create a copy to avoid modifying the original
    styled_df = df.copy()
    
    if col_list is None:
        col_list = styled_df.columns
    
    # Process each column
    for col in col_list:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Get indices of bottom (top) n values in the column
            min_indices = df[col].sort_values(ascending = ascending).iloc[:n].index
            # Apply bold formatting using LaTeX syntax to these values
            for idx in min_indices:
                styled_df.loc[idx, col] = f"\\textbf{{{np.round(styled_df.loc[idx, col],n_digits)}}}"
    
    return styled_df

def latex_display_table_with_std(summary_df, n_highlighted_values : int = 1, ascending:bool = True):
    
    df = summary_df.copy()
    df.index = [m+" - "+f for m,f in zip(df.index.get_level_values(0), df.index.get_level_values(1))]
    outputs = {}
    ths = []
    for time_horizon in df.columns.get_level_values(0).unique():
        if not time_horizon == 'Average':
            min_indices = df[(time_horizon, 'mean')].sort_values(ascending = ascending).iloc[:n_highlighted_values].index
            ths.append(time_horizon)
            for index, row in df.iterrows():
                mean_str = f"{df.loc[index, (time_horizon,'mean')]:.2f}"
                std_str = f"{df.loc[index, (time_horizon,'std')]:.2f}"
                if index in min_indices :
                    display_str = f"\\textbf{{{mean_str} $\pm$ {std_str}}}"
                else :
                    display_str = f"{mean_str} $\pm$ {std_str}"
                try :
                    outputs[index][time_horizon] = display_str
                except :
                    outputs[index] = {time_horizon : display_str}
    output_df = pd.DataFrame(outputs).T.reset_index().rename(columns = {'index':'model'})
    output_df['features'] = [f.split(" ")[-1] for f in output_df['model']]
    output_df['model'] = [" ".join(f.split(" ")[:-2]).replace("_"," ") for f in output_df['model']]
    return output_df[['model','features'] + ths]