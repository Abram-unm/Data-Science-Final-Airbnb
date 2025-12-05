import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

#setup paths
SCRIPT_DIR = Path(__file__).parent 
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUT_DIR = SCRIPT_DIR.parent / 'prepared'
EDA_DIR = SCRIPT_DIR.parent / 'eda_plots'

#features
discreteAttributes  = ["city", "host_type", "zipcode"]
numericalAttributes = ["nightly rate", "bedrooms", "bathrooms", "lead time", 
                       "length stay", "openness", "hot_tub", "pool", "latitude", "longitude"]
target = "high_occupancy"

def save_numeric_distributions(df_, numeric_cols, outdir="eda_plots"):
    os.makedirs(outdir, exist_ok=True)
    existing_cols = [c for c in numeric_cols if c in df_.columns]
    
    #create histogram loop
    for col in existing_cols:
        plt.figure(figsize=(6, 4))
        series = df_[col].dropna()
        plt.hist(series, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"{col} Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{col}_hist.png"))
        plt.close()

#load data
print("Loading Data...")
try:
    market_2019 = pd.read_csv(DATA_DIR / 'market_analysis_2019.csv', sep=';')
    market_general = pd.read_csv(DATA_DIR / 'market_analysis.csv', sep=';')
    amenities = pd.read_csv(DATA_DIR / 'amenities.csv', sep=';')
    geolocation = pd.read_csv(DATA_DIR / 'geolocation.csv', sep=';')
except FileNotFoundError:
    print(f"Error: Files not found in {DATA_DIR}")
    exit()

#consolidate
def clean_id(val): return str(val) if str(val).startswith('AIR') else 'AIR' + str(val)
for df in [market_2019, market_general, amenities, geolocation]:
    df['unified_id'] = df['unified_id'].apply(clean_id)

market_df = pd.concat([market_2019, market_general], ignore_index=True)

#static Features (pool, hot tub)
amenities_static = amenities.groupby('unified_id')[['hot_tub', 'pool']].max().reset_index()
#fix geo commas
geolocation['latitude'] = pd.to_numeric(geolocation['latitude'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
geolocation['longitude'] = pd.to_numeric(geolocation['longitude'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
geo_static = geolocation.groupby('unified_id')[['latitude', 'longitude']].mean().reset_index()

#merge data
full_df = market_df.merge(amenities_static, on='unified_id', how='left')
full_df = full_df.merge(geo_static, on='unified_id', how='left')

#fix european numbers
cols_to_fix = ['occupancy', 'nightly rate', 'lead time', 'length stay', 'bedrooms', 'bathrooms']
for col in cols_to_fix:
    full_df[col] = pd.to_numeric(full_df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

#Target & Date
'''use threshold for classification:

class 1= high occupancy (70% bookings or higher)

class 0= steady-low occupancy (below 70%))

- Drop rows where we don't know occupancy since we cant train

- Convert 'month' format to a Date object to use for splitting 2019-2021 vs 2022 
'''
full_df = full_df.dropna(subset=['occupancy'])
full_df[target] = (full_df['occupancy'] >= 0.70).astype(int)
full_df['date'] = pd.to_datetime(full_df['month'])

#apply One-Hot Encode so that model doesn't break later
#convert text to numbers BEFORE splitting to ensure columns match.

#fill Missing Text with "missing"
for c in discreteAttributes:
    full_df[c] = full_df[c].fillna("missing")

#One-Hot Encode (Get Dummies)
'''This converts 'city' to 'city_BigBear', 'city_JoshuaTree', etc.'''

print("Encoding categorical text to numbers...")
full_df_encoded = pd.get_dummies(full_df, columns=discreteAttributes, drop_first=True)

#SPLIT TRAIN (2019-2021) vs TEST (2022)
print("Splitting Data by Time...")
train_mask = full_df_encoded['date'] < '2022-01-01'
test_mask = full_df_encoded['date'] >= '2022-01-01'

train_df = full_df_encoded[train_mask].copy()
test_df = full_df_encoded[test_mask].copy()

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape:  {test_df.shape}")

#NUMERICS-fill NaN amenities with 0
train_df[['hot_tub', 'pool']] = train_df[['hot_tub', 'pool']].fillna(0)
test_df[['hot_tub', 'pool']]  = test_df[['hot_tub', 'pool']].fillna(0)

#Calculate Medians on TRAIN
continuous_vars = ["nightly rate", "bedrooms", "bathrooms", "lead time", "length stay", "openness"]
num_medians = train_df[continuous_vars].median()
print("\nComputed Medians (on Train):")
print(num_medians)

#apply to both: train and test
train_df[continuous_vars] = train_df[continuous_vars].fillna(num_medians)
test_df[continuous_vars]  = test_df[continuous_vars].fillna(num_medians)

#SAVE
print("\nGenerating EDA Plots...")
save_numeric_distributions(train_df, continuous_vars, outdir=EDA_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

'''drop the non-numeric columns that we didn't encode (i.e. 'month', 'revenue')
- keep all the new dummy columns + numerics + target
'''
cols_to_drop = ['unified_id', 'month', 'revenue', 'occupancy', 'street_name', 'guests', 'date']
train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

train_df.to_csv(OUT_DIR / "train_preprocessed.csv", index=False)
test_df.to_csv(OUT_DIR / "test_preprocessed.csv", index=False)

print(f"\nSuccess! Wrote numeric-ready files to: {OUT_DIR}")
print(f"Columns in final file: {len(train_df.columns)}")