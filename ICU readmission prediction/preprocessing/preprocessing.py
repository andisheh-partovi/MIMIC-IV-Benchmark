import pandas as pd
from sklearn.preprocessing import LabelEncoder,  StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

def imputation(config, df):
    imp_type = config['imp_type']
    if imp_type['mean']:
        df.fillna(df.mean(), inplace=True)
    elif imp_type['median']:
        df.fillna(df.median(), inplace=True)
    else:
        print("MICE Impute is under construction...")
    return df

def outlier_heal(config, data):
    outlier_heal = config[5]['outlier_heal']['byAnomalyScore']
    
    if outlier_heal:
        iforest = IsolationForest(n_estimators=200, max_samples='auto', 
                          contamination=0.03, max_features=.8, 
                          bootstrap=False, n_jobs=-1, random_state=42)
        feats = [c for c in data.columns if c not in ['READMISSION-NTW','READMISSION-30', 'READMISSION-7', 'charttime', 
                                          'subject_id', 'hadm_id', 'stay_id']]
        print(data[feats].shape)

        data.fillna(data.mean(), inplace=True)
        pred= iforest.fit_predict(data[feats])
        data['scores']=iforest.decision_function(data[feats])
        data['original_paper_score']= [-1*s + 0.5 for s in data['scores']]
        data['anomaly_label']=pred

        #print(data[data.anomaly_label==-1].shape)
        data = data[data['original_paper_score']<=0.52]
        drps = ['scores', 'original_paper_score','anomaly_label']
        data.drop(drps, axis=1, inplace=True)
        return data
    else: 
        
        print('just clean data.. Normalization and advanced anomaly healing will presnt on next version ... change config setting ')


def encode(config, frst_stay, categories):
    cat_encoding = config[4]['cat_encoding']['le']
    if cat_encoding:
        le = LabelEncoder()
        for c in categories:
            frst_stay[c] = le.fit_transform(frst_stay[c].astype(str))
        return frst_stay
    #elif cat_encoding== 'woe':
        ### WOE code
    

def missing_rem(df):
    missingTr = df.isnull().sum()/len(df)
    missingTr = missingTr[missingTr>=0]
    missingTr = missingTr.sort_values()
    missingTr = pd.DataFrame({'col':missingTr.index, '%null':missingTr.values})
    drps = missingTr[missingTr['%null']>.95].col.values
    df.drop(drps, axis=1, inplace=True)
    return df
    
def clean_temporals(stays):
    stays = stays[stays['admittime']< stays['dischtime']]
    print("shape after incorrect admit and discharge times:",  stays.shape)
    stays = stays[stays['intime']< stays['outtime']]
    print("shape after incorrect intime and outtimes:",  stays.shape)
    return stays

def clean_events_items(df, event_feats):
    for c in event_feats:
        df = df[(df[c]<999999) |(df[c].isnull())]
        #print(df.describe())
        return df

def prep_stays_dts(stays):
    
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.admittime = pd.to_datetime(stays.admittime)
    stays.dischtime = pd.to_datetime(stays.dischtime)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    #stays.sort_values(by=['intime', 'outtime'], inplace=True)
    return stays

def extract_temporal_feats(frst_stay):
    frst_stay['intime_day']=frst_stay.intime.dt.day
    frst_stay['intime_hour']=frst_stay.intime.dt.hour
    frst_stay['admittime_day']=frst_stay.admittime.dt.day
    frst_stay['admittime_hour']=frst_stay.admittime.dt.hour
    frst_stay['los_hrs'] = frst_stay.los.astype('float32') * 24
    frst_stay['lostostay'] =  frst_stay.intime - frst_stay.admittime
    frst_stay['lostostay']= frst_stay['lostostay'].dt.total_seconds() / 3600
    return frst_stay


    