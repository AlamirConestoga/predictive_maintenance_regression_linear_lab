# Predictive Maintenance with Streaming Data and Linear Regression Alerts
---

## Project Summary

This project simulates real-world predictive maintenance: 
**catching early signs of equipment failure before critical breakdown occurs**.

### Key Workflow
1. **Data Ingestion:** Uploads sensor readings (8 axes of industrial current) to a cloud database ([Neon.tech](https://neon.tech)).
2. **Regression Baseline:** Fits `Time → Axis` univariate linear regressions to serve as stationary baselines.
3. **Threshold Discovery:** Analyzes residual distributions to establish percentile-based `MinC` (Alert) and `MaxC` (Error) thresholds without relying on arbitrary hardcoded limits.
4. **Real-Time Anomaly Detection:** Streams a synthetic test sequence into the database, evaluating continuous deviation durations ($T \ge 5\text{s}$) to flag stateful Alert and Error incidents.
5. **Reproducible Logging:** Logs all flagged events back to PostgreSQL and exports structured CSVs and diagnostic visualizations.

## Setup Instructions

### 1. Clone and enter the repo

```bash
git clone https://github.com/<your-username>/DataStreamVisualization_Workshop.git
cd DataStreamVisualization_Workshop

### 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Configure Neon.tech connection
Create a .env file at the project root (this file must never be committed): DATABASE_URL=postgresql://<user>:<password>@<host>/<dbname>?sslmode=require

How to get the URL:

Sign up at https://neon.tech (free tier).

Create a new project.

Copy the Connection string from the dashboard.

Paste it into .env.

### 5. Run the notebook
Open predictive_maintenance_regression_linear_lab.ipynb in VS Code or Jupyter, select the .venv kernel, and Run cell by cell.

The notebook will:

Connect to Neon.

Upload training data (RMBR4-2_export_test.csv).

Train regression models per axis.

Discover Alert/Error thresholds from residuals.

Generate synthetic test data.

Stream it through the database.

Detect Alert/Error events.

Log events back to Neon and export CSVs.

## Methodology

1. **Regression Baseline:** Fits `Axis_value = slope × time_seconds + intercept` for each axis. Very low $R^2$ values verify that currents remain stationary around a mean, allowing the regression line to serve as a baseline.
2. **Residual Analysis & Thresholds:** Thresholds are calculated non-parametrically from residual distributions (`actual - prediction`):
   * **MinC (Alert):** 95th percentile ($Q_{95}$)
   * **MaxC (Error):** 99th percentile ($Q_{99}$)
   * **T:** 5 seconds continuous duration
3. **Axis Selection:** **Axis #2** is selected for demonstration because of well-behaved residuals ($\text{Std} = 6.88$) and clean $2\times$ threshold separation ($\text{MinC} = 13.55$, $\text{MaxC} = 27.83$).
4. **Stateful Detection:** Tracks continuous deviation duration ($t \ge T$). Once an incident fires, it does not re-fire until values drop below `MinC`, ensuring a 1:1 incident-to-event ratio.

---
## Results

### Discovered Thresholds (Axis #2)

| Threshold | Value |
| :--- | :--- |
| **MinC (Alert)** | 13.55 |
| **MaxC (Error)** | 27.83 |
| **T (Duration)** | 5 s |

### Detected Events

| Axis | Type | Duration | Max Deviation | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Axis #2** | ALERT | 5.0 s | 22.43 | Triggered |
| **Axis #2** | ERROR | 5.0 s | 34.96 | Triggered |

---

## Technologies Used

* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Database:** `PostgreSQL` (Neon.tech), `psycopg2`
* **Visualization:** `matplotlib`
* **Environment:** `python-dotenv`, `Jupyter`

---

## Author & License

* **Course:** Foundation ML — Data Stream Visualization Workshop
* **Author:** Alamir Ibrahim.