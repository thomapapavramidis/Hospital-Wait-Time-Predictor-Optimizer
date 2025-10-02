import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const defaultFormData = {
  arrival_delta_min: 10,
  est_service_time_min: 10,
  lambda_per_min: 0.5,
  mu_per_min: 0.3,
  servers: 2,
  Wq_analytic_min: 18,
  Wq_simulated_480min: 22,
  NumScannersUsedToday: 3,
  InProgressSize: 5,
  SumInProgress: 12,
  NumCompletedToday: 18,
  SumHowEarlyWaiting: 30,
  AvgHowEarlyWaiting: 3,
  LineCount0: 1,
  LineCount1: 1,
  LineCount2: 0,
  LineCount3: 0,
  LineCount4: 0,
  SumWaits: 40,
  DelayCount: 4,
  DelayCountLastHour: 2,
  AvgWaitByTaskTypeLine: 7,
  DelayedInLine: 1,
  is_holiday: false,
  is_event: false,
  dow_sin: 0.78,
  dow_cos: 0.62,
  hour_sin: 0.5,
  hour_cos: 0.87,
  recent_wait_30min: 19,
  recent_wait_60min: 20
};

const FIELD_GROUPS = [
  {
    title: "Arrival & Service",
    description: "Traffic assumptions that feed the queueing model.",
    fields: [
      { name: "arrival_delta_min", label: "Arrival Delta (min)", step: 0.1 },
      { name: "est_service_time_min", label: "Estimated Service Time (min)", step: 0.1 },
      { name: "lambda_per_min", label: "Arrival Rate (per min)", step: 0.01 },
      { name: "mu_per_min", label: "Service Rate (per min)", step: 0.01 },
      { name: "servers", label: "Servers", step: 1, parser: "int" }
    ]
  },
  {
    title: "Queue Analytics",
    description: "Outputs from analytical and simulated wait models.",
    fields: [
      { name: "Wq_analytic_min", label: "Wq Analytic (min)", step: 0.1 },
      { name: "Wq_simulated_480min", label: "Wq Simulated 480 (min)", step: 0.1 }
    ]
  },
  {
    title: "Scanner & Workload",
    description: "Operational stats for scanners and in-progress work.",
    fields: [
      { name: "NumScannersUsedToday", label: "Scanners In Use", step: 1, parser: "int" },
      { name: "InProgressSize", label: "In-Progress Size", step: 1, parser: "int" },
      { name: "SumInProgress", label: "Sum In Progress", step: 0.1 },
      { name: "NumCompletedToday", label: "Completed Today", step: 1, parser: "int" }
    ]
  },
  {
    title: "Queue Status",
    description: "Snapshot of customer counts and delays right now.",
    fields: [
      { name: "SumHowEarlyWaiting", label: "Sum How Early Waiting", step: 0.1 },
      { name: "AvgHowEarlyWaiting", label: "Avg How Early Waiting", step: 0.01 },
      { name: "LineCount0", label: "Line Count 0", step: 1, parser: "int" },
      { name: "LineCount1", label: "Line Count 1", step: 1, parser: "int" },
      { name: "LineCount2", label: "Line Count 2", step: 1, parser: "int" },
      { name: "LineCount3", label: "Line Count 3", step: 1, parser: "int" },
      { name: "LineCount4", label: "Line Count 4", step: 1, parser: "int" },
      { name: "SumWaits", label: "Sum Waits", step: 0.1 },
      { name: "DelayCount", label: "Delay Count", step: 1, parser: "int" },
      { name: "DelayCountLastHour", label: "Delay Count (Last Hour)", step: 1, parser: "int" },
      { name: "AvgWaitByTaskTypeLine", label: "Avg Wait by Task Type Line", step: 0.1 },
      { name: "DelayedInLine", label: "Delayed In Line", step: 1, parser: "int" }
    ]
  },
  {
    title: "Calendar Context",
    description: "Seasonality and special-day indicators.",
    fields: [
      { name: "is_holiday", label: "Is Holiday", type: "checkbox" },
      { name: "is_event", label: "Is Event", type: "checkbox" },
      { name: "dow_sin", label: "Day of Week (sin)", step: 0.01 },
      { name: "dow_cos", label: "Day of Week (cos)", step: 0.01 },
      { name: "hour_sin", label: "Hour (sin)", step: 0.01 },
      { name: "hour_cos", label: "Hour (cos)", step: 0.01 }
    ]
  },
  {
    title: "Recent History",
    description: "Rolling average waits from the last hour.",
    fields: [
      { name: "recent_wait_30min", label: "Recent Wait 30 min", step: 0.1 },
      { name: "recent_wait_60min", label: "Recent Wait 60 min", step: 0.1 }
    ]
  }
];

const parseValue = (value, field) => {
  if (value === "") {
    return "";
  }

  if (field?.parser === "int") {
    return parseInt(value, 10);
  }

  return parseFloat(value);
};

export default function App() {
  const [formData, setFormData] = useState(defaultFormData);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (event, field) => {
    const { name, value, type, checked } = event.target;
    const nextValue = type === "checkbox" ? checked : parseValue(value, field);

    setFormData((prev) => ({
      ...prev,
      [name]: nextValue
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(formData).map(([key, val]) => [
          key,
          typeof val === "string" && val.trim() === "" ? null : val
        ])
      );

      const response = await axios.post("http://127.0.0.1:8000/predict", payload);
      setPrediction(response.data.predicted_wait_time_minutes);
    } catch (err) {
      console.error(err);
      setPrediction(null);
      setError("Prediction request failed. Confirm the API is running and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(defaultFormData);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Wait Time Predictor</h1>
        <p>Fine-tune your operational inputs and estimate the expected wait in minutes.</p>
      </header>

      <main className="content-grid">
        <section className="card form-card">
          <form onSubmit={handleSubmit}>
            {FIELD_GROUPS.map((group) => (
              <fieldset key={group.title} className="form-section">
                <legend>{group.title}</legend>
                <p className="section-subtitle">{group.description}</p>
                <div className="fields-grid">
                  {group.fields.map((field) => (
                    <label
                      key={field.name}
                      className={`field-item${field.type === "checkbox" ? " checkbox" : ""}`}
                    >
                      <span>{field.label}</span>
                      {field.type === "checkbox" ? (
                        <input
                          type="checkbox"
                          name={field.name}
                          checked={Boolean(formData[field.name])}
                          onChange={(event) => handleChange(event, field)}
                        />
                      ) : (
                        <input
                          type="number"
                          inputMode="decimal"
                          step={field.step ?? 0.01}
                          name={field.name}
                          value={formData[field.name]}
                          onChange={(event) => handleChange(event, field)}
                        />
                      )}
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
            <div className="form-actions">
              <button type="submit" className="primary" disabled={isLoading}>
                {isLoading ? "Predicting..." : "Run Prediction"}
              </button>
              <button type="button" onClick={handleReset} className="secondary" disabled={isLoading}>
                Reset Inputs
              </button>
            </div>
          </form>
        </section>

        <section className="card summary-card">
          <h2>Prediction</h2>
          {prediction !== null && !error && (
            <p className="prediction-value">{prediction} minutes</p>
          )}
          {prediction === null && !error && (
            <p className="prediction-placeholder">Submit inputs to see the projected wait.</p>
          )}
          {error && <p className="error-banner">{error}</p>}

          <div className="info-panel">
            <h3>How it works</h3>
            <ul>
              <li>Review and adjust the operational, demand, and calendar inputs.</li>
              <li>Submit the form to call the local FastAPI endpoint backed by the gradient boosting model.</li>
              <li>Use the response to compare scenarios and plan staffing or throughput.</li>
            </ul>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <small>Running locally - Start the FastAPI backend before submitting predictions.</small>
      </footer>
    </div>
  );
}

