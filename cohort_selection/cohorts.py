def get_first_stays(stays):
    frst_stay = stays.sort_values(by=['subject_id', 'intime']).groupby('subject_id').head(1)
    return frst_stay
    
def remove_mortality_frst_stay(frst_stays_data):
    frst_stay = frst_stays_data[~(frst_stays_data.deathtime < frst_stays_data.outtime)]
    return frst_stay

def remove_admission_type(stays, admission_type='Elective'):
    stays = stays[stays.admission_type != admission_type]
    return stays

def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.deathtime.notnull() & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime))
    mortality = mortality | (stays.deathtime.isnull() & stays.dod.notnull() & ((stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays

def add_inunit_mortality_to_icustays(stays):
    mortality = stays.deathtime.notnull() & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime))
    mortality = mortality | (stays.deathtime.isnull() & stays.dod.notnull() & ((stays.intime <= stays.dod) & (stays.outtime >= stays.dod)))

    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays

def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    print(to_keep.head())
    to_keep = to_keep[(to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)][['hadm_id']]
    stays = stays.merge(to_keep, how='inner', left_on='hadm_id', right_on='hadm_id')
    return stays


def filter_icustays_on_age(stays, min_age=18):
    stays = stays[stays.anchor_age > min_age]
    return stays




