import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score
import pygeohash as pgh

# Check if pygeohash is installed, else mock it
try:
    import pygeohash as pgh
    HAS_GEOHASH = True
except ImportError:
    HAS_GEOHASH = False

def preprocess(df, train_stats=None):
    df = df.copy()
    
    # Fill missing
    df['RoadType'] = df['RoadType'].fillna("Unknown")
    df['Weather'] = df['Weather'].fillna("Unknown")
    df['LargeVehicles'] = df['LargeVehicles'].replace({'Allowed': 1, 'Not Allowed': 0, np.nan: 0})
    df['Landmarks'] = df['Landmarks'].replace({'Yes': 1, 'No': 0, np.nan: 0})
    
    # Time features
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True).astype(int)
    df['time_in_mins'] = df['hour'] * 60 + df['minute']
    
    # Geohash decoding
    if HAS_GEOHASH:
        df['lat'] = df['geohash'].apply(lambda x: pgh.decode(x)[0])
        df['lon'] = df['geohash'].apply(lambda x: pgh.decode(x)[1])
    else:
        # Fallback if no pygeohash: label encoding
        pass

    return df

print("Loading data...")
train = pd.read_csv('/home/dell/traffic_demand_prediction/datasets/train.csv')

if not HAS_GEOHASH:
    print("Please install pygeohash for better results. Fallback to Target Encoding.")

# Let's write a proper pipeline and train/test split.
