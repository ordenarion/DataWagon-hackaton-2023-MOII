import sys
import pandas as pd
from catboost import CatBoostClassifier
from wagon_package import prediction_cols

wagnum_lst = sys.argv[1]
forecast_month = sys.argv[1]

month_model_file = 'models/month_prediction_model.cbm'
day_model_file = 'models/day_prediction_model.cbm'
month_model = CatBoostClassifier()
day_model = CatBoostClassifier()
month_model.load_model(str(month_model_file))
day_model.load_model(str(day_model_file))

df = pd.read_csv('evaluation_data/features.csv')
df['reestr_state'] = df['reestr_state'].astype(int)
df_sample = df[(df['wagnum'].isin(wagnum_lst)) & (df['forecast_month'] == forecast_month)]
predictions_month = month_model.predict_proba(df_sample[prediction_cols])[:, 1]
predictions_day = day_model.predict_proba(df_sample[prediction_cols])[:, 1]
pred_df = df_sample[['wagnum']].copy()
pred_df['target_month'] = predictions_month
pred_df['target_day'] = predictions_day

print(pred_df)
