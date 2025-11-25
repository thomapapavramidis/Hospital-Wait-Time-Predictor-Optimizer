import pandas as pd
import numpy as np
import holidays
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
df['dow_sin']  = np.sin(2*np.pi*df['dow']/7)
df['dow_cos']  = np.cos(2*np.pi*df['dow']/7)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

# Recent-history features High Imp!!
# rolling average wait in last 30, 60 minutes
df = df.sort_values('x_ArrivalDTTM')
df['recent_wait_30min'] = df.rolling('30min', on='x_ArrivalDTTM')['Wait'].mean()
df['recent_wait_60min'] = df.rolling('60min', on='x_ArrivalDTTM')['Wait'].mean()


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
    'dow_sin','dow_cos','hour_sin','hour_cos',
    # recent history
    'recent_wait_30min','recent_wait_60min'
]

feature_cols = [c for c in feature_cols if c in df.columns]

df = df.dropna(subset=feature_cols + ['Wait'])


X = df[feature_cols]
y = df['Wait']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Test baseline
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline (predict mean) RMSE: {baseline_rmse:.2f} minutes")
print(f"Baseline (predict mean) MAE:  {baseline_mae:.2f} minutes")

# RF
param_dist = {
    'n_estimators': [100,200,500],
    'max_depth': [None,10,20],
    'min_samples_leaf': [1,5,10],
    'max_features': ['sqrt','log2']
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42,n_jobs=-1),
    param_dist, n_iter=20, cv=3,
    scoring='neg_mean_squared_error',
    random_state=42, verbose=1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
rf_preds = best_rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_medae = median_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)
rf_p90 = np.percentile(np.abs(y_test - rf_preds), 90)
print("RF best params:", rf_search.best_params_)
print(f"RF RMSE: {rf_rmse:.2f} minutes")
print(f"RF MAE:  {rf_mae:.2f} minutes")
print(f"RF Median AE: {rf_medae:.2f} minutes")
print(f"RF 90th percentile AE: {rf_p90:.2f} minutes")
print(f"RF R^2: {rf_r2:.3f}")

# HGB
gb = HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.1, random_state=42
)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
gb_mae = mean_absolute_error(y_test, gb_preds)
gb_medae = median_absolute_error(y_test, gb_preds)
gb_r2 = r2_score(y_test, gb_preds)
gb_p90 = np.percentile(np.abs(y_test - gb_preds), 90)
print(f"HGB RMSE: {gb_rmse:.2f} minutes")
print(f"HGB MAE:  {gb_mae:.2f} minutes")
print(f"HGB Median AE: {gb_medae:.2f} minutes")
print(f"HGB 90th percentile AE: {gb_p90:.2f} minutes")
print(f"HGB R^2: {gb_r2:.3f}")


joblib.dump(best_rf, r"C:\Users\mhdto\OneDrive\Documents\Project DWT\rf_model.joblib")
joblib.dump(gb,     r"C:\Users\mhdto\OneDrive\Documents\Project DWT\gb_model.joblib")
