# Predictive Analytics for Disaster Resource Allocation

Simple AI & Intelligent Systems project for forecasting relief-kit demand in NGO disaster operations.

## Use Case

Forecast **7-day dry ration kit demand** for flood-prone blocks in coastal Odisha. The target users are NGO operations managers who need to decide where to pre-position supplies before monsoon flooding.

The demo uses synthetic but realistic district/block-level data so the project runs offline. It can later be replaced with IMD rainfall, river gauge, NDMA incident, census, and OpenStreetMap road-access data.

## What Is Included

- Synthetic historical dataset for 5 flood-prone blocks.
- Reproducible model training script.
- Accuracy report with MAE, RMSE, and MAPE.
- Feature importance chart.
- 30-day forecast output.
- Streamlit dashboard for non-technical users.
- 1-page NGO resource-allocation playbook.

## Project Structure

```text
disaster_resource_allocation_project/
  app.py
  requirements.txt
  README.md
  data/
    odisha_flood_resource_demand.csv
  models/
    ration_demand_model.joblib
  reports/
    accuracy_report.csv
    feature_importance.csv
    forecast_next_30_days.csv
    resource_allocation_playbook.md
  src/
    generate_data.py
    train_model.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Results

```bash
python src/generate_data.py
python src/train_model.py
```

## Run Dashboard

```bash
streamlit run app.py
```

## Model

The model is a lightweight NumPy ridge-regression forecaster with lag features. It predicts daily dry-ration kit demand, then the dashboard summarizes demand for the next 7 to 30 days.

Main features:

- Rainfall in mm
- River level
- Flood incident count
- Population exposed
- Road accessibility score
- Previous demand lags
- Rolling rainfall and demand averages

## Success Metric

The target benchmark is **MAPE below 20%** on the final 20% of historical dates.
