#------------------------ Header Comment ----------------------#
#train_models.py — Battery RUL (Remaining Useful Life) prediction

import os, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#Clean and prepare the raw DataFrame for model training.
#- Remove NaN / infinite values.
#   - Sort by file and cycle for chronological order.
#   - Compute Remaining Useful Life (RUL) for each sample as:
#         RUL = (max_cycle_per_battery - current_cycle)
#   - Return a subset of essential columns.
#Parameters：
#df : pd.DataFrame   Contains at least three columns: ['file', 'cycle', 'soh' , 'capacity']
#Returns :
#pd.DataFrame   The cleaned data contains ['file', 'cycle', 'soh' , 'rul']
def _clean(df):
    df = df.copy()
    for c in ['cycle','soh','capacity']:
        if c in df.columns:
            df = df[np.isfinite(df[c])]
    df['cycle'] = df['cycle'].astype(int)
    df = df.sort_values(['file','cycle'])
    max_c = df.groupby('file')['cycle'].transform('max')
    df['rul'] = (max_c - df['cycle']).clip(lower=0)
    return df[['file','cycle','soh','rul']]

#Train and evaluate multiple RUL prediction models.
#- Clean dataset with `_clean()`.
#   - Split into training/testing sets (80/20).
#   - Train two regressors:
#       1) Linear Regression
#       2) Random Forest Regressor
#   - Evaluate each model using Mean Absolute Error (MAE) and R² score.
#   - Save models and results summary to disk.
#Parameters：
#df : pd.DataFrame   Contains at least three columns: ['file', 'cycle', 'soh' , 'capacity']
#output_dir : str, optional   Model and result output directory, default is 'outputs'.
#Returns :
#Generates in the output dir:
#   - LinearRegression_model.pkl
#   - RandomForest_model.pkl
#   - results.csv
def train_and_evaluate(df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    dfc = _clean(df)
    if len(dfc) < 20:
        print('⚠️ Not enough data to train.'); return
    X = dfc[['cycle','soh']].values
    y = dfc['rul'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42)
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)
        rows.append((name, mae, r2))
        joblib.dump(m, os.path.join(output_dir, f'{name}_model.pkl'))
        print(f'{name}: MAE={mae:.3f}, R2={r2:.3f}')
    pd.DataFrame(rows, columns=['Model','MAE','R2']).to_csv(os.path.join(output_dir,'results.csv'), index=False)
    print('✅ Training complete. Saved to', output_dir)
