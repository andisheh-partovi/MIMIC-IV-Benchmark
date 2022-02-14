import sys
import os
from os.path import join
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
from tqdm import tqdm
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





def break_up_stays_by_subject(stays, output_path, subjects=None, verbose=1):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays.loc[stays.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'), index=False)
        
    if verbose:
        sys.stdout.write('DONE!\n')




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
    metadata = events[['charttime', 'stay_id']].sort_values(by=['charttime', 'stay_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    tseries = events[['charttime', variable_column, 'valuenum']]\
                    .sort_values(by=['charttime', variable_column, 'valuenum'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    tseries = tseries.pivot(index='charttime', columns=variable_column, values='valuenum').merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in tseries:
            tseries[v] = np.nan
    return tseries

def resample_agg(config, df):
    is_early = config[1]['subtask']['isEarly']
    resagg = []
    for stay_id in tqdm(df.stay_id.unique()):
        s= df[df.stay_id==stay_id]
        s.sort_values(by=['stay_id', 'charttime'], inplace=True)
        s.set_index('charttime', inplace=True)
        
        if is_early:
            s= s.groupby('stay_id',  as_index=False).apply(lambda grp: grp.resample('12H').mean()).reset_index().drop(['level_0'], axis=1).iloc[1:2, :]
            s = s[s['stay_id'].notna()]
            resagg.append(s)
        else: 
            s = s.groupby('stay_id',  as_index=False).apply(lambda grp: grp.resample('24H').mean()).reset_index().drop(['level_0'], axis=1).tail(1)
            s = s[s['stay_id'].notna()]
            resagg.append(s)
                
    master_stays= pd.concat(resagg, axis=0)
    return master_stays
    

def roll_chart_items(config, first_stays):
    from os.path import join
    print('\nImporting data from chartevents...')
    data_dir = config[0]['data_dir']
    print(data_dir)
   
    path_chevents = os.path.join(data_dir, 'chart_events_df.pkl')
    
    chartevents = pd.read_pickle(path_chevents)
    del chartevents['storetime']
    del chartevents['warning']
    del chartevents['value']
    sel_feats = ['SOFA Score','Arterial Blood Pressure diastolic',
       'Arterial Blood Pressure mean', 'Arterial Blood Pressure systolic',
        'Heart Rate','Non Invasive Blood Pressure diastolic',
       'Non Invasive Blood Pressure mean','Non Invasive Blood Pressure systolic', 'Temperature Fahrenheit',
        'charttime', 'Admission Weight (Kg)', 'Admission Weight (lbs.)', 'Apnea Interval', 
         'Called Out', 'Code Status','Cuff Pressure', 'Cuff Volume (mL)', 'Cuff Volume/units', 
         'Daily Weight', 'ETT Location' , 'ETT Mark (cm)' , 'ETT Mark (location)', 'ETT Size (ID)', 'ETT Type', 'Expiratory Ratio',                         
        'Feeding Weight', 'Flow Pattern',  'Flow Rate (variable/fixed)', 'Flow Sensitivity', 'Fspn High',                               
        'Health Care Proxy', 'Height', 'Height (cm)', 'Humidification',                          
        'ICU Consent Signed', 'Inspiratory Ratio', 'Inspiratory Time', 'Inspired O2 Fraction' ,                   
        'Known difficult intubation' , 'Mean Airway Pressure' ,'Minute Volume' ,                          
        'NIV Mask', 'O2 Delivery Device(s)',  'O2 Flow',   'O2 Flow (additional cannula)',  'O2 saturation pulseoxymetry'  ,           
        'PEEP set' , 'PSV Level', 'Paw High', 'Peak Insp. Pressure',                     
        'Plateau Pressure' , 'Respiratory Rate'  ,  'Respiratory Rate (Set)' ,'Respiratory Rate (Total)', 'Respiratory Rate (spontaneous)',          
        'Service'  , 'Slope'  , 'Small Volume Neb Drug #2'   , 'Small Volume Neb Drug/Dose #1'  ,          
        'Sputum Amount'  , 'Sputum Color', 'Sputum Consistency' , 'Sputum Source'   ,                        
        'Tidal Volume (observed)', 'Tidal Volume (set)' ,'Tidal Volume (spontaneous)' , 'Total PEEP Level'  ,                      
        'Ventilator Mode', 'Ventilator Tank #1'  ,  'Ventilator Tank #2'  ,'Ventilator Type'   , 'Vti High, ALT'  ,                                   
        'AST' ,   'Activated Clotting Time',  'Alkaline Phosphate' 'Anion gap',    'Arterial Base Excess',  'Arterial CO2 Pressure', 'Arterial O2 Saturation' ,'Arterial O2 pressure' ,                   
        'BUN'  'CK (CPK)', 'CK-MB' , 'CK-MB fraction (%)' , 'Calcium non-ionized' , 'Chloride (serum)' ,                        
        'Cholesterol' , 'Creatinine (serum)'  ,                                
        'Differential-Basos','Differential-Eos' , 'Differential-Lymphs' , 'Differential-Monos',  'Differential-Neuts' ,                     
        'Glucose (serum)', 'Glucose finger stick (range 70-100)' ,     
        'HCO3 (serum)'  ,'HDL', 'Hematocrit (serum)' , 'Hemoglobin'  , 'INR'   , 'LDL calculated'     ,  'Lactic Acid',                                        
        'Magnesium' , 'PH (Arterial)' ,'PH (dipstick)' ,  'PTT' ,'Phosphorous' , 'Platelet Count',                          
        'Potassium (serum)'  , 'Prothrombin time'  ,  'Sodium (serum)' , 'Specific Gravity (urine)' ,'TCO2 (calc) Arterial'                           
        'Total Bilirubin',  'Triglyceride' , 'Troponin-T' , 'APACHEIII', 'Apache IV A-aDO2', 'ApacheIV_LOS', 
        'Eye Care','Back Care'   ,     'Bath'    ,  'Bed Bath'  ,   'Cough/Deep Breath' , 'APACHE IV diagnosis', 
         'Incentive Spirometry'   ,       'OCAT - Lips Tongue Gums Palate'  ,'OCAT - Saliva secretions', 'Voice quality',                        
        'OCAT - Swallow'  ,   'OCAT - Teeth' ,'Skin Care',   'Subglottal Suctioning' , 'Anti Embolic Device',                                            
        'Anti Embolic Device Status'   ,   'Assistance'    ,   'Assistance Device'   , 'FiO2ApacheIIValue', 'OxygenScore_ApacheIV',                                           
        'PCO2_ApacheIV' ,   'PH_ApacheIV'  , 'PHPaCO2Score_ApacheIV'   ,'PO2_ApacheIV'  , 'RR_ApacheIV'   , 
         'GCS - Eye Opening', 'GCS - Verbal Response','GCS - Motor Response',     
        'Capillary Refill R', 'Capillary Refill L',
        'Non Invasive Blood Pressure diastolic','Arterial Blood Pressure diastolic','Manual Blood Pressure Diastolic Left',
         'Manual Blood Pressure Diastolic Right','Non-Invasive Blood Pressure Alarm - High',    'Non-Invasive Blood Pressure Alarm - Low',
         'Arterial Blood Pressure Alarm - Low',        'Arterial Blood Pressure Alarm - High',   'ART Blood Pressure Alarm - High', 
         'ART Blood Pressure Alarm - Low',  'ART Blood Pressure Alarm Source',
        'Non Invasive Blood Pressure systolic','Arterial Blood Pressure systolic','Manual Blood Pressure Systolic Left','Manual Blood Pressure Systolic Right',
        'Non Invasive Blood Pressure mean', 'Arterial Blood Pressure mean',
        'Glucose (serum)' , 'Glucose finger stick (range 70-100)', 'Glucose (whole blood)', 'Glucose_ApacheIV', 'GlucoseScore_ApacheIV',
        'PAR-Oxygen saturation','Respiratory rate' ,
        'Temperature Fahrenheit', 'Temperature Celsius', 'TemperatureF_ApacheIV', 'Temperature Site', 'Skin Temperature', 
        'PH (Arterial)', 'PH (Venous)', 'PH (dipstick)',  'PHApacheIIValue',  'PH_ApacheIV', 'PHPaCO2Score_ApacheIV', 
        'Admission Weight (lbs.)', 'Daily Weight', 'Feeding Weight',  'Admission Weight (Kg)', 'Weight Bearing Status']
 
    feats =['subject_id', 'hadm_id', 'stay_id', 'intime']
                 
    fstay_chrtevnt = first_stays[feats].merge(chartevents, how='inner', left_on=['subject_id', 'hadm_id', 'stay_id'], right_on=['subject_id', 'hadm_id', 'stay_id'])
    
    print('first stays-chartevents row conts: ', fstay_chrtevnt.shape[0])
    fstay_chrtevnt = fstay_chrtevnt[(fstay_chrtevnt.intime <= fstay_chrtevnt.charttime)]
    path_chartitems = os.path.join(data_dir, 'd_items.csv.gz')
    items =  pd.read_csv(path_chartitems)
    items = items[items.linksto == 'chartevents']
    
    items.drop(['lownormalvalue', 'highnormalvalue'], axis=1, inplace=True)
    fstay_chrtevnt = fstay_chrtevnt.merge(items, how='left', on=['itemid'])
    fstchrt_items = fstay_chrtevnt[fstay_chrtevnt['label'].isin(sel_feats)]
    print('pivoting chartevents features...')
    s = pivot_events(fstchrt_items,variables=sel_feats)
    print('pivot done !')
    s.charttime = pd.to_datetime(s.charttime)
    print('starting resampling and rolling stats...')
    master_stays = resample_agg(config, s)
    print('rolling done !')
    master_stays = clean_events_items(master_stays, sel_feats)
    #display(master_stays.dtypes)
    master_stays.to_pickle('chart_roll.pkl')
    return master_stays
    

    
def roll_lab_items(config, first_stays):
    
    print('\nLoading data from labevents...')
    data_dir = config[0]['data_dir']
    print(data_dir)
   
    path_labevents = os.path.join(data_dir, 'lab_events_df.pkl')
    labevents = pd.read_pickle(path_labevents)
    
    labevents= labevents[labevents['hadm_id'].notna()]
    feats =['subject_id', 'hadm_id', 'stay_id', 'intime']
    fstay_labevnt = first_stays[feats].merge(labevents, how='left', on=['subject_id', 'hadm_id'])
    fstay_labevnt = fstay_labevnt[(fstay_labevnt.intime <= fstay_labevnt.charttime)]
    print('first stays-labevents row counts: ', fstay_labevnt.shape[0])
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
    fs = [c for c in s.columns if c not in ['stay_id', 'charttime']]
    s[fs] = s[fs].apply(pd.to_numeric,errors='coerce',  axis=1)
    print('starting resampling and rolling stats...')
    master_stays = resample_agg(config, s)
    print('rolling done !')
    master_stays = clean_events_items(master_stays, sel_feats_lab)
    master_stays.to_pickle('lab_roll.pkl')
    return master_stays

def measures(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    #plt.figure(figsize=figsize)
    #sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    #if xyplotlabels:
    #    plt.ylabel('True label')
    #    plt.xlabel('Predicted label' + stats_text)
    #else:
   #     plt.xlabel(stats_text)
    
    #if title:
    #    plt.title(title)
        
    return recall, f1_score