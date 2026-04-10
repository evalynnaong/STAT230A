import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
import geopandas as gpd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler

def clean(df): 
    # lowercase no space column names
    rename(df)

def prepare(df, feature_list): 
    standardized = stand(df)
    return standardized
    

def rename(df): 
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

def stand(df, feature_list): 
    scaler = StandardScaler(); 
    scaled_names = [f"{col}_std" for col in feature_list]

    df[scaled_names] = scaler.fit_transform(df[feature_list])

    return df