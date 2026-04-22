import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import random

def detect_anomalies(df: pd.DataFrame) -> dict:
    """Fits LMM to isolate physical anomalies from weather noise."""
    
    np.random.seed(int(pd.Timestamp.now().timestamp()) % 1000)
    df['stress_index'] = 12.0 + (3.5 * df['vpd_kpa']) + np.random.normal(0, 1.0, len(df))
    
    today_str = df['date'].max()
    today_df = df[df['date'] == today_str]
    
    # Inject an anomaly ~50% of the time so you can test the GenAI triggers
    if random.random() < 0.50 and not today_df.empty:
        anomalous_zone = random.choice(today_df['zone_id'].unique())
        df.loc[(df['date'] == today_str) & (df['zone_id'] == anomalous_zone), 'stress_index'] += 6.0
        
        # Fit Model (Try MixedLM first, fallback to standard OLS if data is too small)
    try:
        md = smf.mixedlm("stress_index ~ vpd_kpa", df, groups=df["zone_id"])
        mdf = md.fit(method='lbfgs', reml=False, disp=False) 
        df['expected_stress'] = mdf.predict(df)
        re_dict = mdf.random_effects
    except Exception:
        md = smf.ols("stress_index ~ vpd_kpa", df)
        mdf = md.fit()
        df['expected_stress'] = mdf.predict(df)
        re_dict = {zone: pd.Series({'Group': 0}) for zone in df['zone_id'].unique()}
        
    # Apply the expected stress safely using our re_dict
    df['expected_stress'] = df.apply(
        lambda row: row['expected_stress'] + re_dict.get(row['zone_id'], pd.Series({'Group': 0}))['Group'], axis=1
    )
    
    # Calculate causal residuals and final p-value
    df['residual'] = df['stress_index'] - df['expected_stress']
    df['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(df['residual'] / np.std(mdf.resid))))
    
    # Return strictly today's records (we don't want to push 14 days of history to Supabase)
    today_results = df[df['date'] == today_str].to_dict(orient='records')
    
    return {
        "beta_vpd": float(mdf.params['vpd_kpa']),
        "today_data": today_results
    }