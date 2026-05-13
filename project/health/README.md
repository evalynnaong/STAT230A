# Health Hazards & 911 EMS Calls — San Francisco

Analyze relationships between CalEnviroScreen 4.0 health hazard indicators and 911 medical dispatch call volumes across San Francisco census tracts.

## Pipeline

| Step | File | Description |
|---|---|---|
| 1. EDA | `eda.ipynb` | Exploratory analysis of EMS calls and hazards data |
| 2. Clean | `clean.py` + `clean.ipynb` | Data preparation module (load, spatial join, period aggregation, standardization) |
| 3. Model | `model.ipynb` | Negative Binomial & elastic net regression, seasonal comparison, zipcode aggregation |

## Usage

```bash
./run.sh
```

Or run individual notebooks manually via Jupyter with the `proj_230` kernel.

## Data

- `Fire_Department_and_Emergency_Medical_Services_Dispatched_Calls_for_Service.csv` — 911 dispatch records
- `calenviroscreen40resultsdatadictionary_F_2021.xlsx` — CalEnviroScreen 4.0 hazard scores by census tract
- `Planning_Neighborhood_Groups_Map_20260402.csv` — SF neighborhood boundaries
- `environment.yaml` — Conda environment specification

## Environment

```bash
conda env create -f environment.yaml
conda activate proj_230
```
