wag_cols = [
    'model', 'rod_id', 'gruz', 'cnsi_volumek', 'tara', 'cnsi_probeg_kr', 'cnsi_probeg_dr',
    'kuzov', 'telega', 'tormoz', 'tipvozd', 'tippogl', 'norma_km', 'zavod_build'
]

freight_cols = [
    'isload', 'current_skoroport', 'current_naval', 'current_nasip', 'current_naliv',
    'current_openvagons', 'current_soprovod', 'current_smerz'
]

prediction_cols = [
    'wagnum', 'last_dep', 'last_kap', 'nearest_pl', 'wag_age', 'wag_remains',
    'model', 'rod_id', 'gruz', 'cnsi_volumek', 'cnsi_probeg_dr', 'kuzov', 'telega',
    'tipvozd', 'tippogl', 'norma_km', 'zavod_build', 'gr_probeg', 'por_probeg',
    'tr_rems', 'pr_rems', 'gr_distance', 'por_distance', 'isload', 'current_skoroport',
    'current_naval', 'current_naliv', 'current_openvagons', 'current_soprovod', 'current_smerz',
    'ost_prob', 'reestr_state', 'mileage_all', 'flange_sum', 'rim_sum', 'target_month_lag1', 'target_day_lag1'
]
