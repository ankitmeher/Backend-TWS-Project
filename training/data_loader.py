import os
import pandas as pd
from training.config import DATA_PATH, TARGET

def load_data(path=DATA_PATH):
    """
    Loads the dataset from the specified path and performs initial preprocessing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure processed_price_history_data.csv is in the training/database/ folder.")
        
    df = pd.read_csv(path)
    
    # Basic date processing as requested
    if "date" in df.columns:
        # Use format='mixed' to handle dates both with and without microseconds
        df["date"] = pd.to_datetime(df["date"], format='mixed')
        df = df.sort_values("date").reset_index(drop=True)

    # Split into features and target
    # Ignore 'id' if it's present in the CSV
    drop_cols = ["id", "date", "product_name", TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]

    # XGBoost and some metric logic require numerical labels (0 and 1)
    # Mapping 'buy' -> 1 and 'wait' -> 0
    mapping = {'buy': 1, 'wait': 0}
    y = y.map(mapping).fillna(0).astype(int)
    
    return X, y
