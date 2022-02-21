import sys
import os
from os.path import join
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
from tqdm import tqdm
import gc
sys.path.append(os.path.abspath("../preprocessing"))
from preprocessing import clean_events_items
sys.path.append(os.path.abspath("../data"))

def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

def merge_stays_counts(table1, table2):
    return table1.merge(table2, how='inner', left_on=['hadm_id'], right_on=['hadm_id'])


def add_age_to_icustays(stays):
    stays.intime = pd.to_datetime(stays.intime)
    stays.dod = pd.to_datetime(stays.dod)
    stays.dob = pd.to_datetime(stays.dob)
    #df_icu.intime.describe()
    stays['age'] = (stays.intime - stays.dob).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays.loc[stays.age < 0, 'age'] = 90
    return stays

def break_up_visits_by_subject(visits, output_path, subjects=None, verbose=1):
    subjects = visits.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        visits.loc[visits.subject_id == subject_id].sort_values(by='admittime').to_csv(os.path.join(dn, 'visits.csv'), index=False)

    if verbose:
        sys.stdout.write('Break Up Done!\n')







def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        
    return df




       

def pivot_events(events, variable_column='label', variables=[]):
    metadata = events[['charttime', 'hadm_id']].sort_values(by=['charttime', 'hadm_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    tseries = events[['charttime', variable_column, 'valuenum']]\
                    .sort_values(by=['charttime', variable_column, 'valuenum'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    tseries = tseries.pivot(index='charttime', columns=variable_column, values='valuenum').merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in tseries:
            tseries[v] = np.nan
    gc.collect()
    return tseries


def resample_agg(config, df):
    is_early = config[1]['subtask']['isEarly']
    resagg = []
    for hadm_id in tqdm(df.hadm_id.unique()):
        s= df[df.hadm_id==hadm_id]
        s.sort_values(by=['hadm_id', 'charttime'], inplace=True)
        s.set_index('charttime', inplace=True)
        
        if is_early:
            s= s.groupby('hadm_id',  as_index=False).apply(lambda grp: grp.resample('12H').mean()).reset_index().drop(['level_0'], axis=1).iloc[1:2, :]
            s = s[s['hadm_id'].notna()]
            resagg.append(s)
        else: 
            s = s.groupby('hadm_id',  as_index=False).apply(lambda grp: grp.resample('24H').mean()).reset_index().drop(['level_0'], axis=1).tail(1)
            s = s[s['hadm_id'].notna()]
            resagg.append(s)
                
    master_stays= pd.concat(resagg, axis=0)
    gc.collect()
    return master_stays




    

    
def roll_lab_items(config, first_stays):
    
    print('\nLoading data from labevents...')
    data_dir = config[0]['data_dir']
    print(data_dir)
   
    path_labevents = os.path.join(data_dir, 'labevents.csv.gz')
    labevents = pd.read_csv(path_labevents)
    labevents= labevents[labevents['hadm_id'].notna()]
   
    feats =['subject_id', 'hadm_id','admittime']
    fstay_labevnt = first_stays[feats].merge(labevents, how='left', on=['subject_id', 'hadm_id'])
    fstay_labevnt =fstay_labevnt[fstay_labevnt.labevent_id.notnull()]
    
    print('\nremove labevents out of visit los....')
    fstay_labevnt = fstay_labevnt[(fstay_labevnt.admittime <= fstay_labevnt.charttime)]
    path_labitems = os.path.join(data_dir, 'd_labitems.csv.gz')
    items =pd.read_csv(path_labitems)
    f1 = list(items[items.category=='Blood Gas']['label'].unique())
    f2 = list(items[items.category=='Chemistry']['label'].unique())
    f3 = list(items[items.category=='Hematology']['label'].unique())
    sel_feats_lab=f1+f2+f3
    chart_items = fstay_labevnt.merge(items, how='left', on=['itemid'])
    
    print('pivoting labevents features...')
    s = pivot_events(chart_items,variables=sel_feats_lab)
    print('pivot done !')
    s.charttime = pd.to_datetime(s.charttime)
    #fs = [c for c in s.columns if c not in ['stay_id', 'charttime']]
    #s[fs] = s[fs].apply(pd.to_numeric,errors='coerce',  axis=1)
    print('starting resampling and rolling stats...')
    master_stays = resample_agg(config, s)
    print('rolling done !')
    print('cleaning laevents items values ...')
    master_stays = clean_events_items(master_stays, sel_feats_lab)
    print('Unique hadm_id: ', master_stays.hadm_id.nunique())
    master_stays.to_pickle('lab_roll.pkl')
    return master_stays

