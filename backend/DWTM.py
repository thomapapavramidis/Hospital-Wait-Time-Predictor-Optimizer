import pandas as pd
import numpy as np
import holidays
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
import joblib

csv_path = r"C:\Users\mhdto\OneDrive\Documents\Project DWT\enriched_wait_data.csv"
df = pd.read_csv(
    csv_path,
    parse_dates=['x_ArrivalDTTM','x_ScheduledDTTM','x_BeginDTTM']
)

# Basic datetime & arrival/service features ===
# arrival_delta_min, est_service_time_min, lambda_per_min, mu_per_min, servers, Wq_analytic_min, Wq_simulated_480min,
# lambda_hat_per_hr, exp_scale_min, weibull_shape, weibull_scale

df['sched_date'] = df['x_ScheduledDTTM'].dt.normalize()
us_hols = holidays.US()
df['is_holiday'] = df['sched_date'].isin(us_hols)

# fixed-date holidays + Thanksgiving/Black Friday
fixed_events = {
    "New Year": "01-01",
    "Valentine": "02-14",
    "StPatrick": "03-17",
    "July4th": "07-04",
    "Halloween": "10-31",
    "XmasEve": "12-24",
    "Xmas": "12-25",
}
def nth_weekday(year, month, weekday, n):
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7*(n-1))

years = df['sched_date'].dt.year.unique()
events = []
for y in years:
    for name, mmdd in fixed_events.items():
        events.append({'date': pd.to_datetime(f"{y}-{mmdd}").normalize(),
                       'event': name})
    thx = nth_weekday(y,11,weekday=3,n=4).normalize()  # Thanksgiving
    events.append({'date': thx, 'event':'Thanksgiving'})
    events.append({'date': thx + pd.Timedelta(days=1), 'event':'BlackFriday'})

events_df = pd.DataFrame(events)
df = df.merge(events_df, left_on='sched_date', right_on='date', how='left')
df['is_event'] = df['event'].notna()
df.drop(columns=['date','event'], inplace=True)

# Queue-status features 
queue_cols = [
    'SumHowEarlyWaiting','AvgHowEarlyWaiting',
    'LineCount0','LineCount1','LineCount2','LineCount3','LineCount4',
    'SumWaits','DelayCount','DelayCountLastHour',
    'AvgWaitByTaskTypeLine','DelayedInLine'
]
# Gotta make sure all queue_cols are in df; drop any missing
queue_cols = [c for c in queue_cols if c in df.columns]

#Cyclical time features
df['dow']      = df['x_ScheduledDTTM'].dt.dayofweek
df['hour']     = df['x_ScheduledDTTM'].dt.hour
df['hour_of_week'] = df['dow'] * 24 + df['hour']
df['dow_sin']  = np.sin(2*np.pi*df['dow']/7)
df['dow_cos']  = np.cos(2*np.pi*df['dow']/7)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['how_sin']  = np.sin(2*np.pi*df['hour_of_week']/168)
df['how_cos']  = np.cos(2*np.pi*df['hour_of_week']/168)

# Recent-history features: rolling averages over past 30/60 minutes.
df = df.sort_values('x_ArrivalDTTM')
df['recent_wait_30min'] = df.rolling('30min', on='x_ArrivalDTTM')['Wait'].mean()
df['recent_wait_60min'] = df.rolling('60min', on='x_ArrivalDTTM')['Wait'].mean()
# Additional short/long lags and trend
df['recent_wait_15min'] = df.rolling('15min', on='x_ArrivalDTTM')['Wait'].mean()
df['recent_wait_120min'] = df.rolling('120min', on='x_ArrivalDTTM')['Wait'].mean()
df['last_wait'] = df['Wait'].shift(1)
df['recent_wait_trend'] = df['recent_wait_30min'] - df['recent_wait_120min']


feature_cols = [
    # arrival & service
    'arrival_delta_min','est_service_time_min','lambda_per_min','mu_per_min',
    'servers','Wq_analytic_min','Wq_simulated_480min',
    # scanner & demographic
    'NumScannersUsedToday','InProgressSize','SumInProgress','NumCompletedToday',
    # queue-status
] + queue_cols + [
    # holiday/event
    'is_holiday','is_event',
    # cyclical time
    'dow_sin','dow_cos','hour_sin','hour_cos','how_sin','how_cos',
    # recent history
    'recent_wait_15min','recent_wait_30min','recent_wait_60min','recent_wait_120min','recent_wait_trend','last_wait'
]

feature_cols = [c for c in feature_cols if c in df.columns]

# Drop useless rows
df = df.dropna(subset=feature_cols + ['Wait'])


# Chronological split for lookahead leakage
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Signed log transform for early admits cause they're negative
def signed_log1p(v):
    return np.sign(v) * np.log1p(np.abs(v))

def signed_expm1(v):
    return np.sign(v) * np.expm1(np.abs(v))


X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
y_train_raw = train_df['Wait']
y_test_raw = test_df['Wait']

# Transform for modeling; cap extremes to reduce variance, then invert after prediction.
y_train_capped = y_train_raw.clip(lower=-180, upper=180)
y_test_capped = y_test_raw.clip(lower=-180, upper=180)  # metrics still use uncapped y_test_raw
y_train = signed_log1p(y_train_capped)
y_test = signed_log1p(y_test_capped)

# Just metric helper
def summarize(name, y_true, y_pred):
    err = y_pred - y_true
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    p90 = np.percentile(np.abs(err), 90)
    r2 = r2_score(y_true, y_pred)
    mean_err = np.mean(err)
    median_err = np.median(err)
    sign_match = np.mean(np.sign(y_true) == np.sign(y_pred))
    pos_mask = y_true > 0
    neg_mask = y_true < 0
    mae_pos = mean_absolute_error(y_true[pos_mask], y_pred[pos_mask]) if pos_mask.any() else np.nan
    mae_neg = mean_absolute_error(y_true[neg_mask], y_pred[neg_mask]) if neg_mask.any() else np.nan

    print(f"{name} RMSE: {rmse:.2f} minutes")
    print(f"{name} MAE:  {mae:.2f} minutes")
    print(f"{name} Median AE: {medae:.2f} minutes")
    print(f"{name} 90th percentile AE: {p90:.2f} minutes")
    print(f"{name} R^2: {r2:.3f}")
    print(f"{name} Mean Error (bias): {mean_err:.2f} minutes")
    print(f"{name} Median Error: {median_err:.2f} minutes")
    print(f"{name} Sign agreement (pred/actual): {sign_match*100:.1f}%")
    if pos_mask.any():
        print(f"{name} MAE on delays (Wait>0): {mae_pos:.2f} minutes")
    if neg_mask.any():
        print(f"{name} MAE on early admits (Wait<0): {mae_neg:.2f} minutes")
    print("-" * 50)

# Baseline on original scale (AM).
baseline_pred = np.full_like(y_test_raw, y_train_raw.mean(), dtype=float)
summarize("Baseline (mean)", y_test_raw, baseline_pred)

# RF
def estimate_gap_rows(df, gap_minutes=60):
    """Approximate a row-count gap equivalent to gap_minutes to purge lookahead."""
    deltas = df['x_ArrivalDTTM'].diff().dt.total_seconds().dropna() / 60
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= 0:
        return 1
    return max(1, int(np.ceil(gap_minutes / median_delta)))


rf_param_dist = {
    'n_estimators': [200,400,800],
    'max_depth': [None,12,20],
    'min_samples_leaf': [5,10,20],
    'min_samples_split': [2,5,10],
    'max_features': ['sqrt','log2', 0.3, 0.5, 0.7]
}

gap_rows = estimate_gap_rows(train_df, gap_minutes=60)
tscv = TimeSeriesSplit(n_splits=3, gap=gap_rows)
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42,n_jobs=-1),
    rf_param_dist, n_iter=25, cv=tscv,
    scoring='neg_mean_absolute_error',  # focus on MAE to reduce typical error
    random_state=42, verbose=1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
rf_preds_log = best_rf.predict(X_test)
rf_preds = signed_expm1(rf_preds_log)
print("RF best params:", rf_search.best_params_)
summarize("RF", y_test_raw, rf_preds)

# HGB tuned via randomized search with early stopping and regularization
hgb_param_dist = {
    'learning_rate': [0.02, 0.05, 0.08],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0.0, 0.1, 0.3, 1.0],
    'max_iter': [400, 800, 1200]
}
hgb_search = RandomizedSearchCV(
    HistGradientBoostingRegressor(
        early_stopping=True,
        random_state=42
    ),
    hgb_param_dist,
    n_iter=25,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    verbose=1
)
hgb_search.fit(X_train, y_train)
gb = hgb_search.best_estimator_
gb_preds_log = gb.predict(X_test)
gb_preds = signed_expm1(gb_preds_log)
print("HGB best params:", hgb_search.best_params_)
summarize("HGB", y_test_raw, gb_preds)

# Quantile HGB for prediction intervals
quantile_alphas = [0.1, 0.5, 0.9]
quantile_models = {}
for alpha in quantile_alphas:
    q_model = HistGradientBoostingRegressor(
        loss='quantile',
        quantile=alpha,
        max_iter=1000,
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=47,
        min_samples_leaf=30,
        l2_regularization=0.2,
        early_stopping=True,
        random_state=42
    )
    q_model.fit(X_train, y_train)
    quantile_models[alpha] = q_model

q_preds = {alpha: signed_expm1(model.predict(X_test)) for alpha, model in quantile_models.items()}
median_preds = q_preds[0.5]
summarize("Quantile-HGB median", y_test_raw, median_preds)

joblib.dump(best_rf, r"C:\Users\mhdto\OneDrive\Documents\Project DWT\rf_model.joblib")
joblib.dump(gb,     r"C:\Users\mhdto\OneDrive\Documents\Project DWT\gb_model.joblib")
joblib.dump(quantile_models, r"C:\Users\mhdto\OneDrive\Documents\Project DWT\gb_quantiles.joblib")
