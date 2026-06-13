# 1-Page NGO Resource-Allocation Playbook

## Use Case

Forecast dry-ration kit demand for flood-prone blocks in coastal Odisha for the next 7 to 30 days.

## Decision Rule

- If forecast upper-bound demand exceeds available local stock, pre-position supplies within 48 hours.
- Use the dashboard safety-buffer slider to add 10-25% extra kits when road access is poor or rainfall uncertainty is high.
- Treat the upper forecast estimate as the planning number when heavy rain and high river levels happen together.

## Daily Workflow

1. Open the Streamlit dashboard every morning during monsoon season.
2. Select the block and planning horizon.
3. Check total kits needed, peak demand day, and model uncertainty range.
4. Compare recommended kits with current warehouse stock.
5. Move supplies first to blocks with high forecast demand and low road accessibility.
6. Record actual distribution after the event so the model can be retrained.

## Features That Matter

The model considers rainfall, river level, flood incidents, exposed population, road access, seasonal timing, and recent demand lags. Feature importance is shown in the dashboard so coordinators can see why demand is rising.

## Handling Data Gaps

- If rainfall is missing, use the nearest IMD station or district average.
- If river gauge data is missing, use rainfall lag and incident reports as proxy signals.
- If road access is unknown, start with a neutral score of 0.5 and update it after field validation.
- Keep a manual override column for local staff observations.

## Communicating Uncertainty

Use three numbers: likely demand, lower estimate, and upper estimate. For operational decisions, the coordinator should plan against the upper estimate when transport delays are likely.

## Next Real-Data Upgrade

Replace synthetic data with IMD rainfall, NDMA incident logs, district census population, OpenStreetMap road distance, and NGO warehouse stock records.

