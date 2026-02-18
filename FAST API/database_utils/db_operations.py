"""
Database operations for fetching price history
"""
import sqlite3
import pandas as pd
from config import DB_PATH


def get_price_history(product_name: str) -> pd.DataFrame:
    """
    Fetch price history for a given product from the database.
    
    Args:
        product_name (str): The name of the product to fetch history for
        
    Returns:
        pd.DataFrame: DataFrame containing price history with columns:
                     date, product_name, brand, price, has_anc, 
                     anc_level_db, has_enc, driver_size_mm
                     
    Raises:
        Exception: If database connection fails or query execution fails
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT
            date,
            product_name,
            brand,
            price,
            has_anc,
            anc_level_db,
            has_enc,
            driver_size_mm
        FROM price_history
        WHERE product_name = ?
        ORDER BY date
        """
        df = pd.read_sql(query, conn, params=(product_name,))
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Error fetching price history: {e}")
        raise
