import pandas as pd
import numpy as np

from wagon_package import wag_cols, freight_cols

path_to_train = '../Documents/wagon/train/'
path_to_test = '../Documents/wagon/test/'

dislocation = pd.read_parquet(path_to_train + 'dislok_wagons.parquet')
dislocation_test = pd.read_parquet(path_to_test + 'dislok_wagons.parquet')
dislocation = dislocation.append(dislocation_test)
pr_rems = pd.read_parquet(path_to_train + 'pr_rems.parquet')
pr_rems_test = pd.read_parquet(path_to_test + 'pr_rems.parquet')
pr_rems = pr_rems.append(pr_rems_test)
tr_rems = pd.read_parquet(path_to_train + 'tr_rems.parquet')
tr_rems_test = pd.read_parquet(path_to_test + 'tr_rems.parquet')
tr_rems = tr_rems.append(tr_rems_test)
wag_prob = pd.read_parquet(path_to_train + 'wagons_probeg_ownersip.parquet')
wag_prob_test = pd.read_parquet(path_to_test + 'wagons_probeg_ownersip.parquet')
wag_prob = wag_prob.append(wag_prob_test)
kti_izm = pd.read_parquet(path_to_train + 'kti_izm.parquet')
kti_izm_test = pd.read_parquet(path_to_test + 'kti_izm.parquet')
kti_izm = kti_izm.append(kti_izm_test)

freight = pd.read_parquet(path_to_train + 'freight_info.parquet')
wag_params = pd.read_parquet(path_to_train + 'wag_params.parquet')

target = pd.read_csv(path_to_train + 'target/y_train.csv')
target_test = pd.read_csv(path_to_test + 'target/y_test.csv')
target = target.append(target_test)

wag_params = wag_params.set_index('wagnum')
freight = freight.set_index('fr_id')
kti_izm['operation_date_dttm'] = pd.to_datetime(kti_izm['operation_date_dttm'])
kti_izm['operation_date_dttm'] = kti_izm['operation_date_dttm'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
flange_cols = [col for col in kti_izm.columns if 'flange' in col]
rim_cols = [col for col in kti_izm.columns if 'rim' in col]

target['month'] = pd.to_datetime(target['month'])
target = target.set_index(['wagnum', 'month'])

# last repairs
dislocation['current_month'] = dislocation['plan_date'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
dislocation['date_dep'] = dislocation['date_dep'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
dislocation['date_kap'] = dislocation['date_kap'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
dislocation['date_pl_rem'] = dislocation['date_pl_rem'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)

# mileage
gr = dislocation[dislocation['isload'] == 1]
por = dislocation[dislocation['isload'] == 0]
gr_distance = gr.groupby(['wagnum', 'current_month'])['distance'].mean()
por_distance = por.groupby(['wagnum', 'current_month'])['distance'].mean()

wag_prob['repdate'] = wag_prob['repdate'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
wag_prob_ost = wag_prob.groupby(['wagnum', 'repdate'])['ost_prob'].min()
wag_prob_reestr = wag_prob.groupby(['wagnum', 'repdate'])['reestr_state'].first()

# freight
for col in freight.columns:
    freight.rename({col: 'current_' + col}, axis=1, inplace=True)
dislocation = dislocation.set_index('fr_id')
dislocation[freight.columns] = freight
dislocation = dislocation.reset_index()
freight_mean = dislocation.groupby(['wagnum', 'current_month'])[freight_cols].mean()

# repairs
pr_rems['rem_month'] = pr_rems['rem_month'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
tr_rems['rem_month'] = tr_rems['rem_month'] + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
pr_rems = pr_rems.sort_values(by=['rem_month'])
tr_rems = tr_rems.sort_values(by=['rem_month'])
tr_rems_cs = tr_rems.groupby(['wagnum', 'rem_month'])['kod_vrab'].count().groupby(level=0).cumsum()
pr_rems_cs = pr_rems.groupby(['wagnum', 'rem_month'])['kod_vrab'].count().groupby(level=0).cumsum()
mla_before_rep = tr_rems.groupby('wagnum')[['gr_probeg', 'por_probeg']].mean()

# flanges, rim
kti_izm['flange_sum'] = kti_izm[flange_cols].sum(axis=1)
kti_izm['rim_sum'] = kti_izm[rim_cols].sum(axis=1)
kti_izm_gr = kti_izm.groupby(['wagnum', 'operation_date_dttm'])[['mileage_all', 'flange_sum', 'rim_sum']].mean()

# aggregate features
df = dislocation[
    ['wagnum', 'current_month', 'date_dep', 'date_kap', 'date_pl_rem']
].drop_duplicates(subset=['wagnum', 'current_month'])

df['last_dep'] = (df['current_month'] - df['date_dep']) / np.timedelta64(1, 'M')
df['last_kap'] = (df['current_month'] - df['date_kap']) / np.timedelta64(1, 'M')
df['nearest_pl'] = (df['date_pl_rem'] - df['current_month']) / np.timedelta64(1, 'M')
df['last_dep'].fillna(1.2 * df['last_dep'].max(), inplace=True)
df['last_kap'].fillna(1.2 * df['last_kap'].max(), inplace=True)
df.drop(['date_dep', 'date_kap', 'date_pl_rem'], axis=1, inplace=True)

df = df.set_index('wagnum')
df[['date_build', 'srok_sl']] = wag_params[['date_build', 'srok_sl']]
df['wag_age'] = (df['current_month'] - df['date_build']) / np.timedelta64(1, 'Y')
df['wag_remains'] = (df['srok_sl'] - df['current_month']) / np.timedelta64(1, 'D')
df.drop(['date_build', 'srok_sl'], axis=1, inplace=True)

df[wag_cols] = wag_params[wag_cols]

df['norma_km'] = df['norma_km'].apply(lambda x: 1 if x == 160000 else (2 if x == 0 else 0))
df['cnsi_probeg_kr'] = df['cnsi_probeg_kr'].apply(lambda x: 1 if x == 110 else 0)
df['cnsi_probeg_dr'] = df['cnsi_probeg_dr'].apply(lambda x: 1 if x == 110 else 0)

df['tippogl'].fillna(-1, inplace=True)
df[mla_before_rep.columns] = mla_before_rep

df = df.reset_index().set_index(['wagnum', 'current_month'])
df['tr_rems'] = tr_rems_cs
df['pr_rems'] = pr_rems_cs
df['gr_distance'] = gr_distance
df['por_distance'] = por_distance
df[freight_cols] = freight_mean
df['ost_prob'] = wag_prob_ost
df['reestr_state'] = wag_prob_reestr
df[kti_izm_gr.columns] = kti_izm_gr

df.fillna(0, inplace=True)

df = df.reset_index()
df['forecast_month'] = df['current_month'] + pd.offsets.MonthBegin(1)
df = df.set_index(['wagnum', 'forecast_month'])
df['target_month'] = target['target_month']
df['target_day'] = target['target_day']

df['prev_month'] = df['current_month'] - pd.offsets.MonthBegin(1)
df = df.reset_index().set_index(['wagnum', 'prev_month'])
df['target_month_lag1'] = target['target_month']
df['target_day_lag1'] = target['target_day']

df.to_csv('evaluation_data/features.csv')
