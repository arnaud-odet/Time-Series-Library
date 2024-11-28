import os
import pandas as pd

log_folder = './logs'

def log(metrics, args):
    
    task = args.task_name

    # Check if the directory exists, if not, create it
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    logs =  pd.concat([pd.DataFrame([vars(args)]), pd.DataFrame([metrics])], axis = 1)

    # Check if the file exists, if not, create it as a .csv with pandas
    filepath = os.path.join(log_folder, task + '.csv')
    if os.path.exists(filepath):
        log_df = pd.read_csv(filepath, index_col=0)
        logs = pd.concat([log_df,logs], ignore_index=True)
    logs.to_csv(filepath)
    
    print(f"logs saved in file {filepath}")
    
    return