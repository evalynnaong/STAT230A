import pandas as pd
import numpy as np
import re
from scipy.spatial import cKDTree


PERIOD_MONTHS = {
    'nov_jan': (11, 12, 1),
    'feb_apr': (2, 3, 4),
    'may_jul': (5, 6, 7),
    'aug_oct': (8, 9, 10),
}

POLLUTANT_COLS = [
    'ozone', 'pm2.5', 'diesel_pm', 'drinking_water', 'lead',
    'pesticides', 'tox._release', 'traffic', 'cleanup_sites',
    'groundwater_threats', 'haz._waste', 'imp._water_bodies', 'solid_waste',
]

HEALTH_COLS = [
    'asthma', 'low_birth_weight', 'cardiovascular_disease',
]

DEMOGRAPHIC_COLS = [
    'education', 'linguistic_isolation', 'poverty', 'unemployment', 'housing_burden',
]

HAZARD_COVARIATES = POLLUTANT_COLS + HEALTH_COLS + DEMOGRAPHIC_COLS


def rename_columns(df):
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]


def load_ems_data(path):
    df = pd.read_csv(path)
    rename_columns(df)
    return df


def load_hazards_data(path):
    sheets = pd.read_excel(path, sheet_name=None)
    hazards = sheets['CES4.0FINAL_results']
    demographics = sheets['Demographic Profile']
    dictionary = sheets['Data Dictionary']
    rename_columns(hazards)
    rename_columns(demographics)
    rename_columns(dictionary)
    return hazards, demographics, dictionary


def filter_medical_incidents(df):
    return df[df['call_type'] == 'Medical Incident'].copy()


def filter_san_francisco(hazards):
    return hazards[hazards['california_county'] == 'San Francisco'].copy()


def parse_wkt_point(point_str):
    if pd.isna(point_str):
        return np.nan, np.nan
    match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(point_str))
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return lat, lon
    return np.nan, np.nan


def extract_coords(df, location_col='case_location'):
    df = df.copy()
    coords = df[location_col].apply(parse_wkt_point)
    df['_latitude'] = coords.apply(lambda x: x[0])
    df['_longitude'] = coords.apply(lambda x: x[1])
    return df


def assign_nearest_tract(med_inc, communities, med_lat_col='_latitude',
                          med_lon_col='_longitude', comm_lat_col='latitude',
                          comm_lon_col='longitude', tract_col='census_tract'):
    communities_clean = communities.dropna(subset=[comm_lat_col, comm_lon_col]).copy()
    med_inc_out = med_inc.copy()

    def to_radians_array(lat, lon):
        return np.radians(np.column_stack([lat, lon]))

    comm_coords_rad = to_radians_array(
        communities_clean[comm_lat_col].values,
        communities_clean[comm_lon_col].values,
    )
    tree = cKDTree(comm_coords_rad)

    med_coords_rad = to_radians_array(
        med_inc_out[med_lat_col].values,
        med_inc_out[med_lon_col].values,
    )

    valid_mask = ~(np.isnan(med_inc_out[med_lat_col].values)
                   | np.isnan(med_inc_out[med_lon_col].values))

    nearest_idx = np.full(len(med_inc_out), -1, dtype=int)
    _, nearest_idx[valid_mask] = tree.query(med_coords_rad[valid_mask], k=1)

    tract_values = communities_clean[tract_col].values
    med_inc_out['census_tract'] = np.where(
        nearest_idx >= 0, tract_values[nearest_idx], np.nan
    )
    return med_inc_out


def assign_period(df):
    df = df.copy()
    df['date_column'] = pd.to_datetime(df['call_date'], format='%m/%d/%Y')
    df['month'] = df['date_column'].dt.month
    df['iso_week'] = df['date_column'].dt.isocalendar().week.astype(int)

    def _period_label(m):
        for label, months in PERIOD_MONTHS.items():
            if m in months:
                return label
        return None

    df['period'] = df['month'].map(_period_label)
    return df


def aggregate_calls_by_period(df):
    weekly = (
        df.groupby(['census_tract', 'period', 'iso_week'])
        .size()
        .reset_index(name='calls_that_week')
    )

    period_stats = (
        weekly.groupby(['census_tract', 'period'])['calls_that_week']
        .agg(['mean', 'std'])
        .reset_index()
    )
    period_stats['mean'] = period_stats['mean'].round(2)
    period_stats['std'] = period_stats['std'].round(2)

    mean_wide = period_stats.pivot(index='census_tract', columns='period', values='mean')
    mean_wide.columns = [f'call_avg_{c}' for c in mean_wide.columns]

    std_wide = period_stats.pivot(index='census_tract', columns='period', values='std')
    std_wide.columns = [f'call_std_{c}' for c in std_wide.columns]

    result = mean_wide.join(std_wide).reset_index()
    result['census_tract'] = result['census_tract'].astype(int)
    return result


def merge_hazards_and_calls(hazards_sf, call_metrics):
    merged = hazards_sf.merge(call_metrics, on='census_tract', how='inner')
    return merged.dropna(subset=HAZARD_COVARIATES)


def standardize(df, feature_list):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_names = [f"{col}_std" for col in feature_list]
    df[scaled_names] = scaler.fit_transform(df[feature_list])
    return df, scaled_names


def prepare_analysis_data(ems_path, hazards_path):
    hazards, demographics, dictionary = load_hazards_data(hazards_path)
    hazards_sf = filter_san_francisco(hazards)

    ems = load_ems_data(ems_path)
    med_inc = filter_medical_incidents(ems)
    med_inc = extract_coords(med_inc, location_col='case_location')
    med_inc = assign_nearest_tract(med_inc, hazards_sf)
    med_inc = assign_period(med_inc)

    call_metrics = aggregate_calls_by_period(med_inc)
    df = merge_hazards_and_calls(hazards_sf, call_metrics)
    df, scaled_cols = standardize(df, HAZARD_COVARIATES)
    print(f"Standardized columns: {scaled_cols}")
    return df
