from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X):
    """
    Creates a ColumnTransformer to handle categorical and numerical features.
    Now includes StandardScaler to help models like Logistic Regression converge.
    """
    cat_cols = ["brand"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
