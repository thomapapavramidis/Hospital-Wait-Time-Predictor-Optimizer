# Project DWT: Wait-Time Modeling for Real Operations

This repo is a project of mine for predicting patient wait time from live queue conditions.
I added a feature-heavy modeling first, then a small FastAPI service, then a React UI for scenario testing, plus visualization tooling for explaining system behavior.


## What This Predicts

The target is `Wait` (minutes), where:
- Positive values mean delayed start.
- Negative values mean early admit.

Current data snapshot (`enriched_wait_data.csv`):
- `82,001` rows
- `100` columns
- Arrival range: `2022-10-30` to `2025-08-02`
- Wait range: `-497` to `360` minutes
- Mean wait: `6.98` minutes
- 90th percentile wait: `39` minutes
- Negative waits: `35.22%` of records

## Repo Map

```text
Project DWT/
  backend/
    DWTM.py                 # training + feature engineering + model export
    app.py                  # FastAPI inference service
    visualizations.py       # heatmap, simulation, phase diagram generators
    gb_model.joblib         # model currently loaded by API (31-feature version)
    hgb_model.joblib        # alternate model artifact
    wait_time_model.joblib  # earlier random forest artifact
    visualizations/         # generated PNG/HTML artifacts
  wait-time-ui/
    src/App.js              # input form + API calls
    src/App.css             # UI styling
    package.json            # CRA app dependencies/scripts
  enriched_wait_data.csv    # feature-enriched training data
  requirements.txt          # Python dependencies
```

## End-to-End Flow

1. Historical data is read from `enriched_wait_data.csv`.
2. `backend/DWTM.py` builds time, queue, and recency features.
3. Multiple regressors are tuned and compared with time-aware CV.
4. Model artifacts are persisted with `joblib`.
5. `backend/app.py` serves `/predict` with a loaded gradient boosting model.
6. `wait-time-ui/src/App.js` sends user-edited operational inputs to the API.
7. `backend/visualizations.py` produces explainability visuals for congestion and stability behavior.

## Modeling Approach (What Is Actually Implemented)

`backend/DWTM.py` includes:
- Holiday + event flags (US holidays, Thanksgiving, Black Friday, and fixed-date events).
- Cyclical encodings for day-of-week/hour/hour-of-week (`sin/cos`).
- Queue state features (line counts, delays, aggregate waits, workload stats).
- Rolling recency features over 15/30/60/120 minute windows.
- Chronological split (`80/20`) to avoid lookahead leakage.
- `TimeSeriesSplit` with a computed row-gap to purge near-future contamination.
- Signed log transform of target after clipping to `[-180, 180]`.
- Randomized hyperparameter search for:
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`
- Quantile HGB models for interval-style predictions (`0.1`, `0.5`, `0.9`).

## Model Artifacts Currently in Repo

These are from different training generations, so treat them as lineage, not one single canonical release.

| File | Estimator | `n_features_in_` | Notes |
|---|---|---:|---|
| `backend/gb_model.joblib` | HistGradientBoostingRegressor | 31 | Loaded by API today |
| `gb_model.joblib` | HistGradientBoostingRegressor | 37 | Newer training output from `DWTM.py` |
| `rf_model.joblib` | RandomForestRegressor | 37 | Newer training output |
| `gb_quantiles.joblib` | dict of HGB models | 37 | Quantile models (`0.1/0.5/0.9`) |
| `xgb_model.joblib` | XGBRegressor | 42 | Experimental/legacy |
| `stack_model.joblib` | StackingRegressor | 42 | Experimental/legacy |
| `backend/hgb_model.joblib` | HistGradientBoostingRegressor | 42 | Older backend artifact |
| `backend/wait_time_model.joblib` | RandomForestRegressor | 8 | Earliest simple model |

## Local Runbook

### 1. Backend (FastAPI)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

API should be live at:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

### 2. Frontend (React)

```powershell
cd wait-time-ui
npm install
npm start
```

UI opens at `http://localhost:3000`.

### 3. Visualization Suite

```powershell
cd backend
python visualizations.py
```

Expected outputs include:
- `backend/visualizations/wait_time_heatmap.html`
- `backend/visualizations/wait_time_heatmap_static.png`
- `backend/visualizations/phase_diagram.png`
- `backend/visualizations/phase_diagram_interactive.html`

##  Things I underdeveloped

- Paths are hard-coded to a local machine in:
  - `backend/DWTM.py`
  - `backend/app.py`
  - `backend/visualizations.py`
- API and latest training script are on different feature sets:
  - API model uses 31 inputs (`backend/gb_model.joblib`)
  - latest training outputs use 37 inputs (`gb_model.joblib`)
- React app calls `http://127.0.0.1:8000/predict`, but backend does not currently configure CORS middleware.
- `wait-time-ui/src/App.test.js` is still the default CRA test and does not match the current UI.
- `backend/visualizations.py` calls `create_queue_simulation_animation(...)` positionally in a way that leaves `save_path` at default (so output location is not what `generate_all_visualizations()` implies).

## Practical Next Steps

1. Replace hard-coded absolute paths with config/env-based relative paths.
2. Version the feature schema and enforce it at training and inference time.
3. Decide one canonical model artifact and remove stale variants from serving path.
4. Add CORS middleware and an API smoke test.
5. Add lightweight regression tests around input schema and inference output type/range.

## Why This Project Is Useful

This is not just a model that outputs a number.
It is a queue-aware decision support prototype that ties:
- operations research intuition (`lambda`, `mu`, stability),
- modern tabular ML practice (feature engineering + time-safe validation),
- and practical delivery (API + UI + explainability visuals).

That combination is where most real operations ML work actually lives.
