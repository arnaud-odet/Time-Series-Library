import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
from data_provider.data_factory import data_provider

LOG_PATH = './logs'
DATA_PATH = "./dataset/USC"
START_INDEX = 232
BASIS_ARGS_LOGS = 207
CALMIP_START_INDEX = {
    2: 471,
    }

def logs_summary(seq_len:int, sort_col:str = 'fde', run : int = 2):
    pred_len = seq_len
    model_id = f'USC_{seq_len}_{pred_len}'    

    # Calmip Logs
    cp_logs_df = pd.read_csv(os.path.join(LOG_PATH,'calmip_logs.csv'), index_col = 0)
    cp_logs_df['fit_time'] = cp_logs_df['fit_time'].astype(int)
    cp_query = (cp_logs_df['model_id'] == model_id) & (cp_logs_df.index >= CALMIP_START_INDEX[run])
    usc_args = SimpleNamespace(**cp_logs_df.loc[CALMIP_START_INDEX[run]].to_dict())

    # Scoring baselines
    usc_args.seq_len = seq_len
    usc_args.pred_len = pred_len
    usc_args.root_path = DATA_PATH
    usc_args.scale = False
    uscds, uscdl = data_provider(args= usc_args, flag = 'test', )
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
    cv_fde, cv_rmse
    ### Concat
    bls = [{'model':'Baseline - StandStill', 'features':'S','fde':ss_fde, 'rmse':ss_rmse},
        {'model':'Baseline - ConstantVelocity', 'features':'S','fde':cv_fde, 'rmse': cv_rmse}]
    columns = ['model','features','fde','rmse']
    tmp_df = cp_logs_df[cp_query][columns].copy().reset_index(drop=True)
    tmp_df.sort_values(by = 'fde', inplace=True)
    tmp_df['dupl'] = tmp_df[['model','features']].duplicated()
    tmp_df=tmp_df[tmp_df['dupl'] == False]

    comp = pd.concat([tmp_df.drop(columns = 'dupl'), pd.DataFrame(bls)]).reset_index(drop=True)
    return comp.sort_values(by = sort_col)


def logs_summary_run_1(seq_len:int, sort_col:str = 'fde'):
    pred_len = seq_len
    model_id = f'USC_{seq_len}_{pred_len}'    

    # Calmip Logs
    cp_logs_df = pd.read_csv(os.path.join(LOG_PATH,'calmip_logs.csv'), index_col = 0)
    cp_logs_df['fit_time'] = cp_logs_df['fit_time'].astype(int)
    cp_query = (cp_logs_df['model_id'] == model_id) & (cp_logs_df.index < CALMIP_START_INDEX[2])


    # Local logs
    lt_logs_df = pd.read_csv(os.path.join(LOG_PATH,'long_term_forecast.csv'), index_col = 0)
    lt_logs_df['epoch'] = lt_logs_df['epoch'].fillna(-1).astype(int)
    lt_logs_df['fit_time'] = np.round(lt_logs_df['fit_time'],2)
    loc_query = (lt_logs_df['model_id'] == model_id)&(lt_logs_df['fde'].isna() == False)&(lt_logs_df.index >= START_INDEX)

    # Scoring baselines
    usc_args = SimpleNamespace(**lt_logs_df.loc[BASIS_ARGS_LOGS].to_dict())
    usc_args.seq_len = seq_len
    usc_args.pred_len = pred_len
    usc_args.root_path = DATA_PATH
    usc_args.scale = False
    uscds, uscdl = data_provider(args= usc_args, flag = 'test', )
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
    cv_fde, cv_rmse
    ### Concat
    bls = [{'model':'Baseline - StandStill', 'features':'S','fde':ss_fde, 'rmse':ss_rmse},
        {'model':'Baseline - ConstantVelocity', 'features':'S','fde':cv_fde, 'rmse': cv_rmse}]
    columns = ['model','features','fde','rmse']
    tmp_df = pd.concat([lt_logs_df[loc_query][columns].copy(), cp_logs_df[cp_query][columns].copy()]).reset_index(drop=True)
    tmp_df.sort_values(by = 'fde', inplace=True)
    tmp_df['dupl'] = tmp_df[['model','features']].duplicated()
    tmp_df=tmp_df[tmp_df['dupl'] == False]

    comp = pd.concat([tmp_df.drop(columns = 'dupl'), pd.DataFrame(bls)]).reset_index(drop=True)
    return comp.sort_values(by = sort_col)

def exp_summary(exp_list, sort_col:str = 'fde', run : int = 1):
    for i, seq_len in enumerate(exp_list):
        if run == 1 :
            tdf = logs_summary_run_1(seq_len, sort_col=sort_col)
        else :
            tdf = logs_summary(seq_len, sort_col=sort_col, run=run)    
        tdf = tdf.set_index(['model','features'])
        cs = [[f"{seq_len}_{seq_len}"], ["fde", "rmse"]]
        c_index = pd.MultiIndex.from_product(cs, names=["seq_len", "metrics"])
        tdf = tdf.set_axis(c_index, axis = 1)
        if i == 0 :
            rdf = tdf.copy()
        else :
            rdf = rdf.merge(tdf, right_index = True, left_index = True, how = 'outer')
    
    metrics = rdf.columns.get_level_values(1).unique()
    for metric in metrics:
        new_col = ('Average', metric)
        rdf[new_col] = rdf.xs(metric, axis=1, level=1).mean(axis=1)
           
    return rdf.reset_index().sort_values(by = ('Average',sort_col))
