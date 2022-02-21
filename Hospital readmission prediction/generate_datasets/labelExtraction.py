import numpy as np
import pandas as pd
from tqdm import tqdm
import sys, os
import gc
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



sys.path.append(os.path.abspath("../cohort_selection"))
from cohorts import add_inhospital_mortality_to_visits
import timeit

def label_extraction():
    start = timeit.default_timer()
    print("\nStarting  label extraction... , grab a coffee !")
    start = timeit.default_timer()
    for subject_dir in tqdm(os.listdir('erVisits/')):
        
        dn = os.path.join('erVisits/', subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
       

        try:
            stays = pd.read_csv(os.path.join('erVisits/', subject_dir, 'visits.csv'))
            #display(stays.head())
        except:
            sys.stdout.write('error reading from disk!\n')
            continue
        else:
            sys.stdout.flush()
            
        stays = add_inhospital_mortality_to_visits(stays)
       
               
        next_admittime = stays[stays.groupby(['hadm_id'])['dischtime'].transform(max) == stays['dischtime']]
        next_admittime = next_admittime[['hadm_id', 'admittime', 'dischtime']]
        next_admittime['NEXT_ADMITTIME'] = next_admittime.admittime.shift(-1)
        next_admittime['DIFF'] =  pd.to_datetime(next_admittime.NEXT_ADMITTIME) -  pd.to_datetime(stays.dischtime)
        stays = stays.merge(next_admittime[['hadm_id', 'DIFF']], how='inner', left_on=['hadm_id'], right_on=['hadm_id'])

        less_than_30days = stays.DIFF.notnull() & (stays.DIFF < '30 days 00:00:00')
        less_than_7days = stays.DIFF.notnull() & (stays.DIFF < '7 days 00:00:00')
        NTW = stays.DIFF.notnull() & (stays.DIFF > '0 days 00:00:00')

        stays['LESS_TAHN_30DAYS'] = less_than_30days.astype(int)
        stays['LESS_TAHN_7DAYS'] = less_than_7days.astype(int)
        stays['NTW'] = NTW.astype(int)
       
        stays['READMISSION-30'] = (stays.LESS_TAHN_30DAYS==1).astype(int)
        stays['READMISSION-7'] = (stays.LESS_TAHN_7DAYS==1).astype(int)
        stays['READMISSION-NTW'] = (stays.NTW==1).astype(int)
        
        stays.to_csv(os.path.join('erVisits/', subject_dir, 'visits_readmission.csv'), index=False)
       
    sys.stdout.write(' Label Extraction DONE!\n')
        
    sys.stdout.write(' Buiding master_visits dataframe  ...\n')
    gc.collect()
    
    master_df = []
    for subject_dir in tqdm(os.listdir('erVisits/')):
        dn = os.path.join('erVisits/', subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
       
        try:
            stays = pd.read_csv(os.path.join('erVisits/', subject_dir, 'visits_readmission.csv'))
            master_df.append(stays)
        except:
            sys.stdout.write('error reading from disk!\n')
            continue
          
    master_visits = pd.concat(master_df, axis=0)
    #drops = ['LESS_TAHN_30DAYS', 'LESS_TAHN_7DAYS','NTW', 'MAX_OUTTIME','edregtime','edouttime','anchor_year','dod',
    #            'hospital_expire_flag', 'COUNTS', 'DIFF' ]
    #master_stays.drop(drops, axis=1, inplace=True)
    sys.stdout.write(' Buiding Master visits dataframe... Finished!\n')
    gc.collect()
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    return master_visits


