"""Run this at the end of the capstone notebook after compare_df is created.
It exports the notebook's hybrid scoring output in the column contract expected by the Streamlit app.
"""
import pandas as pd

raw = loans_df.loc[compare_df['loan_index']].copy()
scored = raw.copy()
for c in ['risk_score','risk_tier','if_anomaly','xgb_proba','iso_score','xgb_rank','iso_rank','y_true']:
    if c in compare_df:
        scored[c] = compare_df[c].values
scored['scoring_mode'] = 'Notebook-exported XGBoost-anchored + IF signal'
scored.to_csv('ppp_dashboard_scored.csv', index=False)
print('Wrote ppp_dashboard_scored.csv')
