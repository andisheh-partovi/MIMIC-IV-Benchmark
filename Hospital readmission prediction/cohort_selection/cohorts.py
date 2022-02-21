def get_first_visit(master_visits):
    frst_vsts = master_visits.sort_values(by=['subject_id', 'admittime']).groupby('subject_id').head(1)
    return frst_vsts
    
def remove_mortality_visits(frst_vsts):
    
    #frst_vsts = frst_vsts[frst_vsts.deathtime.notnull() & ((frst_vsts.admittime <= frst_vsts.deathtime) & (frst_vsts.dischtime >= frst_vsts.deathtime))]
    
    mortality = frst_vsts.deathtime.notnull() & ((frst_vsts.admittime <= frst_vsts.deathtime) & (frst_vsts.dischtime >= frst_vsts.deathtime))
    mortality = mortality | (frst_vsts.deathtime.isnull() & frst_vsts.dod.notnull() & ((frst_vsts.admittime <= frst_vsts.dod) & (frst_vsts.dischtime >= frst_vsts.dod)))
    frst_vsts['MORTALITY_INHOSPITAL'] = mortality.astype(int)
    frst_vsts = frst_vsts[frst_vsts.MORTALITY_INHOSPITAL==0]
    print('mortality counts in first visits :', frst_vsts[frst_vsts.MORTALITY_INHOSPITAL==1].shape[0])
    return frst_vsts

def remove_admission_type(stays, admission_type='Elective'):
    stays = stays[stays.admission_type != admission_type]
    return stays

def add_inhospital_mortality_to_visits(vst):
    mortality = vst.deathtime.notnull() & ((vst.admittime <= vst.deathtime) & (vst.dischtime >= vst.deathtime))
    mortality = mortality | (vst.deathtime.isnull() & vst.dod.notnull() & ((vst.admittime <= vst.dod) & (vst.dischtime >= vst.dod)))
    
    vst['MORTALITY_INHOSPITAL'] = mortality.astype(int)
    return vst



def get_admissions_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    #print(to_keep.head())
    #to_keep = to_keep[(to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)][['hadm_id']]
    #stays = stays.merge(to_keep, how='inner', left_on='hadm_id', right_on='hadm_id')
    return to_keep


def filter_visit_on_age(vsts, min_age=18):
    vsts = vsts[vsts.anchor_age > min_age]
    return vsts




