import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
from typing import Dict, Any, Tuple
import joblib
from colorama import Fore, Style

# CONFIG
DEBUG: bool = True
SUCCESS = Fore.GREEN + Style.BRIGHT
ERROR = Fore.RED + Style.BRIGHT
MAGENTA = Fore.MAGENTA + Style.BRIGHT
CYAN = Fore.CYAN + Style.BRIGHT
RESET = Style.RESET_ALL


class LatencyPredictionModel:
    """Regression model to predict latency"""

    def __init__(self, train_file_path: str = None, test_file_path: str = None):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train_df = None
        self.test_df = None
        self.base_models = None
        self.latency_scaler = joblib.load("../data/Scaler/latency_scaler.pkl")

    def load_preprocessed_dataset(self, file_path: str = None) -> pd.DataFrame:
        """Load preprocessed dataset from a CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print(f"{ERROR}Error: File '{file_path}' not found.{RESET}")
            return None

    def train_val_split(
        self, df: pd.DataFrame = None, ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the dataset into training 85% training and 15% and validation sets."""
        X = df.drop(columns=["latency"])
        y = df["latency"]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=ratio, random_state=46
        )
        return X_train, X_valid, y_train, y_valid

    def train_base_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Trains multiple base models:
        LGBMRegressor, XGBRegressor,
        RandomForestRegressor, CatBoostRegressor.
        """
        models = {
            "LGBM": LGBMRegressor(random_state=46, verbose=-1),
            "XGBoost": XGBRegressor(random_state=46),
            "RandomForest": RandomForestRegressor(random_state=46),
            "CatBoost": CatBoostRegressor(
                random_state=46, silent=True, allow_writing_files=False
            ),
        }

        self.base_models = {}

        for name, model in tqdm(models.items(), desc="Training Base Models"):
            # progress bar
            model.fit(X_train, y_train)
            # get validation r² for training
            train_r2 = r2_score(y_train, model.predict(X_train))
            print(f"{SUCCESS}{name} model R² on training set:   {train_r2:.4f}{RESET}")
            self.base_models[name] = model

        return self.base_models

    def calculate_validation_r2(
        self, X_valid: pd.DataFrame = None, y_valid: pd.Series = None
    ) -> None:
        """
        Calculates R² score for each model on the validation set.
        """
        print()
        for name, model in self.base_models.items():
            val_r2 = r2_score(y_valid, model.predict(X_valid))
            print(f"{MAGENTA}{name} model R² on validation set:   {val_r2:.4f}{RESET}")

    def predict_base_models(self, X_valid: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates predictions using trained base models.
        """
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X_valid)

        ensembled_predictions = pd.DataFrame(predictions)
        return ensembled_predictions

    def test_averaged_model(
        self, save_preds: bool = False, original_test_df: pd.DataFrame = None
    ):
        """
        Tests the trained base models on new data and averages their predictions.
        """
        try:
            test_df = self.load_preprocessed_dataset(self.test_file_path)
            base_predictions = self.predict_base_models(test_df)
            # the resulting prediction is the mean of the base model predictions
            final_predictions = base_predictions.mean(axis=1)
            # unscale the predictions
            unscaled_latency = self.latency_scaler.inverse_transform(
                final_predictions.values.reshape(-1, 1)
            ).flatten()
            predictions_df = test_df.copy()
            predictions_df["predicted_latency"] = unscaled_latency.astype(float)
            final_df = original_test_df.copy()
            final_df["predicted_latency"] = predictions_df["predicted_latency"]
            if save_preds:
                final_df.to_csv("../data/predictions_averaged.csv", index=False)
            print(f"\n{CYAN}Predictions saved to 'predictions_averaged.csv'.{RESET}")

        except Exception as e:
            print(f"{ERROR}Error occurred during prediction: {e}{RESET}")

    def run(self):
        """Pipeline wrapper"""
        self.train_df = self.load_preprocessed_dataset(self.train_file_path)
        X_train, X_valid, y_train, y_valid = self.train_val_split(self.train_df)
        self.train_base_models(X_train, y_train)
        self.calculate_validation_r2(X_valid, y_valid)
        original_test_df = self.load_preprocessed_dataset("../data/testing_data.csv")
        self.test_averaged_model(save_preds=False, original_test_df=original_test_df)


def main():
    train_file_path = "../data/preprocessed_training_data.csv"
    test_file_path = "../data/preprocessed_testing_data.csv"
    model = LatencyPredictionModel(train_file_path, test_file_path)
    model.run()


if __name__ == "__main__":
    main()
