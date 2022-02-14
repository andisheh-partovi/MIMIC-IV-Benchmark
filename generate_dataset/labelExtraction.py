import numpy as np
import pandas as pd
from tqdm import tqdm
import sys, os
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from IPython.display import display
import gc

sys.path.append(os.path.abspath("../utils"))
from utils import  merge_stays_counts, break_up_stays_by_subject


        
def label_extraction():
    
    print("\nStarting  label extraction... , grab a coffee !")
    for subject_dir in tqdm(os.listdir('erStays/')):
        
        dn = os.path.join('erStays/', subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
        
        sys.stdout.flush()

        try:
            
            stays = pd.read_csv(os.path.join('erStays/', subject_dir, 'stays.csv'))
            
            #display(stays.head())
        except:
            sys.stdout.write('error reading from disk!\n')
            continue
        else:
            #sys.stdout.write(
            #    'got {0} stays...'.format(stays.shape[0]))
            sys.stdout.flush()

        #stays = add_inhospital_mortality_to_icustays(stays)
        #stays = add_inunit_mortality_to_icustays(stays)


        counts = stays.groupby(['hadm_id']).size().reset_index(name='COUNTS')
        stays = merge_stays_counts(stays, counts)
        max_outtimme = stays.groupby(['hadm_id'])['outtime'].transform(max) == stays['outtime']
        stays['MAX_OUTTIME'] = max_outtimme.astype(int)
        transferback = (stays.COUNTS > 1) & (stays.MAX_OUTTIME == 0)
        stays['TRANSFERBACK'] = transferback.astype(int)
        
        next_admittime = stays[stays.groupby(['hadm_id'])['outtime'].transform(max) == stays['outtime']]
        next_admittime = next_admittime[['hadm_id', 'stay_id', 'admittime', 'dischtime']]
        next_admittime['NEXT_ADMITTIME'] = next_admittime.admittime.shift(-1)
        next_admittime['DIFF'] =  pd.to_datetime(next_admittime.NEXT_ADMITTIME) -  pd.to_datetime(stays.dischtime)
        stays = merge_stays_counts(stays, next_admittime[['hadm_id', 'DIFF']])

        less_than_30days = stays.DIFF.notnull() & (stays.DIFF < '30 days 00:00:00')
        less_than_7days = stays.DIFF.notnull() & (stays.DIFF < '7 days 00:00:00')
        NTW = stays.DIFF.notnull() & (stays.DIFF > '0 days 00:00:00')

        stays['LESS_TAHN_30DAYS'] = less_than_30days.astype(int)
        stays['LESS_TAHN_7DAYS'] = less_than_7days.astype(int)
        stays['NTW'] = NTW.astype(int)
       
        stays['READMISSION-30'] = ((stays.TRANSFERBACK==1) |(stays.LESS_TAHN_30DAYS==1)).astype(int)
        stays['READMISSION-7'] = ((stays.TRANSFERBACK==1) |(stays.LESS_TAHN_7DAYS==1)).astype(int)
        stays['READMISSION-NTW'] = ((stays.TRANSFERBACK==1) |(stays.NTW==1)).astype(int)
        
        stays.to_csv(os.path.join('erStays/', subject_dir, 'stays_readmission.csv'), index=False)

    sys.stdout.write(' Label Extraction DONE!\n')
        
    sys.stdout.write(' Buiding Master Stays dataframe, about 200 seconds...\n')
    
    master_df = []
    for subject_dir in tqdm(os.listdir('erStays/')):
        dn = os.path.join('erStays/', subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
       
        try:
            stays = pd.read_csv(os.path.join('erStays/', subject_dir, 'stays_readmission.csv'))
            master_df.append(stays)
        except:
            sys.stdout.write('error reading from disk!\n')
            continue
          
    master_stays = pd.concat(master_df, axis=0)
    #drops = ['LESS_TAHN_30DAYS', 'LESS_TAHN_7DAYS','NTW', 'MAX_OUTTIME','edregtime','edouttime','anchor_year','dod',
    #            'hospital_expire_flag', 'COUNTS', 'DIFF' ]
    #master_stays.drop(drops, axis=1, inplace=True)
    sys.stdout.write(' Master Stays df... DONE!\n')
    return master_stays