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
    3: np.inf
    }

# Utils

def create_model_identifier(row):
    return f"{row['model']}_{row['features']}_{row['seq_len']}-{row['pred_len']}_el{row['e_layers']}_dm{row['d_model']}_nh{row['n_heads']}_dl{row['d_layers']}_ff{row['d_ff']}"

def score_baselines(args, seq_len : int, pred_len : int):
    # Scoring baselines
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
    bls = [{'model':'Baseline - StandStill', 'features':'S','fde':ss_fde, 'rmse':ss_rmse},
        {'model':'Baseline - ConstantVelocity', 'features':'S','fde':cv_fde, 'rmse': cv_rmse}]
    
    return pd.DataFrame(bls)


# Logs

def mean_std_logs_summary(metric:str = 'fde', run : int = 2):

    # Calmip Logs loading and filtering
    cp_logs_df = pd.read_csv(os.path.join(LOG_PATH,'calmip_logs.csv'), index_col = 0)
    cp_query = (cp_logs_df.index >= CALMIP_START_INDEX[run]) & (cp_logs_df.index < CALMIP_START_INDEX[run+1])
    cp_logs_df = cp_logs_df[cp_query]
    
    # Processing logs
    cp_logs_df['model_id'] = cp_logs_df.apply(lambda row : create_model_identifier(row), axis =1)
    cp_logs_df['time'] = cp_logs_df.apply(lambda row : f"{row['seq_len']}-{row['pred_len']}", axis =1)
    temp_df = cp_logs_df[['model_id','model','features','time','rmse',metric]].groupby('model_id').agg(
        model=pd.NamedAgg(column='model', aggfunc='first'),
        features=pd.NamedAgg(column='features', aggfunc='first'),
        time=pd.NamedAgg(column='time', aggfunc='first'),
        count=pd.NamedAgg(column= metric, aggfunc='count'),
        mean=pd.NamedAgg(column= metric, aggfunc='mean'),
        std=pd.NamedAgg(column= metric, aggfunc='std'),
    ).sort_values(by = 'mean')
    temp_df['dupl'] = temp_df[['model','time','features']].duplicated()
    temp_df['std'] = temp_df['std'].fillna(0)
    summary_df = temp_df[temp_df['dupl'] == False].copy()
    summary_df = summary_df.pivot(index = ['model','features'], columns = 'time', values = ['mean','std']).swaplevel(0,1,axis = 1).sort_index(axis = 1)
    
    # Scoring baselines
    summary_df.loc[('Baseline - StandStill', 'S'),:] = np.nan
    summary_df.loc[('Baseline - ConstantVelocity', 'S'),:] = np.nan
    usc_args = SimpleNamespace(**cp_logs_df.loc[CALMIP_START_INDEX[run]].to_dict())  
    for time_horizon in summary_df.columns.get_level_values(0).unique():
        seq_len, pred_len = time_horizon.split('-')[0],time_horizon.split('-')[1]     
        bls = score_baselines(usc_args, seq_len=seq_len, pred_len=pred_len)
        summary_df.loc[('Baseline - StandStill', 'S'), (time_horizon, 'mean')] = bls[bls['model'] == 'Baseline - StandStill'][metric].iloc[0]
        summary_df.loc[('Baseline - StandStill', 'S'), (time_horizon, 'std')] = 0
        summary_df.loc[('Baseline - ConstantVelocity', 'S'), (time_horizon, 'mean')] = bls[bls['model'] == 'Baseline - ConstantVelocity'][metric].iloc[0]
        summary_df.loc[('Baseline - ConstantVelocity', 'S'), (time_horizon, 'std')] = 0
           
    summary_df[('Average','mean')] = summary_df.xs('mean', axis = 1, level = 1).mean(axis = 1)
    summary_df.sort_values(by = ('Average','mean'),inplace=True)
    
    return summary_df
    
def logs_summary(seq_len:int, sort_col:str = 'fde', run : int = 2):
    pred_len = seq_len
    model_id = f'USC_{seq_len}_{pred_len}'    

    # Calmip Logs
    cp_logs_df = pd.read_csv(os.path.join(LOG_PATH,'calmip_logs.csv'), index_col = 0)
    cp_logs_df['fit_time'] = cp_logs_df['fit_time'].astype(int)
    cp_query = (cp_logs_df['model_id'] == model_id) & (cp_logs_df.index >= CALMIP_START_INDEX[run]) & (cp_logs_df.index < CALMIP_START_INDEX[run+1])
    usc_args = SimpleNamespace(**cp_logs_df.loc[CALMIP_START_INDEX[run]].to_dict())

    bls = score_baselines(usc_args, seq_len=seq_len, pred_len=pred_len)

    columns = ['model','features','fde','rmse']
    tmp_df = cp_logs_df[cp_query][columns].copy().reset_index(drop=True)
    tmp_df.sort_values(by = 'fde', inplace=True)
    tmp_df['dupl'] = tmp_df[['model','features']].duplicated()
    tmp_df=tmp_df[tmp_df['dupl'] == False]

    comp = pd.concat([tmp_df.drop(columns = 'dupl'), bls]).reset_index(drop=True)
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
    bls = score_baselines(usc_args, seq_len=seq_len, pred_len=pred_len)

    columns = ['model','features','fde','rmse']
    tmp_df = pd.concat([lt_logs_df[loc_query][columns].copy(), cp_logs_df[cp_query][columns].copy()]).reset_index(drop=True)
    tmp_df.sort_values(by = 'fde', inplace=True)
    tmp_df['dupl'] = tmp_df[['model','features']].duplicated()
    tmp_df=tmp_df[tmp_df['dupl'] == False]

    comp = pd.concat([tmp_df.drop(columns = 'dupl'), bls]).reset_index(drop=True)
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