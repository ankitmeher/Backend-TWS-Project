from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    """
    Returns a dictionary of models and their hyperparameter grids for GridSearchCV.
    """
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {
                "model__C": [0.1, 1, 10]
            }
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced"),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [5, 10]
            }
        ),
        "xgboost": (
            XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6],
                "model__learning_rate": [0.05, 0.1]
            }
        )
    }
