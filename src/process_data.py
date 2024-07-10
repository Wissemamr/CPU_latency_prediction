from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Optional, Literal
from sklearn.preprocessing import LabelEncoder
import joblib
from colorama import Fore, Style


# CONFIG
SUCCESS = Fore.GREEN + Style.BRIGHT
ERROR = Fore.RED + Style.BRIGHT
VERBOSE = Fore.CYAN + Style.BRIGHT
MAGENTA = Fore.MAGENTA + Style.BRIGHT
RESET = Fore.RESET
SAVE: bool = False


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        assert file_path is not None
    except AssertionError:
        raise ValueError("File path cannot be None")
    return pd.read_csv(file_path)


def drop_id(df: pd.DataFrame, col: str = "ID"):
    """Drop ID column"""
    try:
        df = df.drop(columns=[col])
        return df
    except Exception as e:
        print(f"Error dropping ID : {e}")


def encode_ram_limit(df: pd.DataFrame, col: str = "ram_limit"):
    """transform ram limit to int"""
    try:
        df[col] = df[col].str.replace("M", "").astype(int)
        return df
    except Exception as e:
        print(f"Error encoding ram limit : {e}")


def encode_cpu_type(df: pd.DataFrame, col: str = "cpu_type"):
    """Encode cpu type as brand, model series and model number"""
    try:
        pattern = r"(\w+)\s([\w\s]+)\s([\w\-]+)"
        df[["brand", "model_series", "model_number"]] = df["cpu_type"].str.extract(
            pattern
        )
        df = df.drop(columns=["cpu_type"])
        # encode brand , model_series and model_number with label encoding
        categorical_cols = ["brand", "model_series", "model_number"]
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
        return df
    except Exception as e:
        print(f"Error in encoding cpu type : {e}")


def convert_timestamp(df: pd.DataFrame, col: str = "timestamp"):
    """Convert timestamp to datetime and extract day_of_week and time_of_day"""
    try:
        df[col] = pd.to_datetime(df[col])
        df["day_of_week"] = df[col].dt.day_name().astype("category").cat.codes
        df_ = df.drop(columns=[col])
        return df_
    except Exception as e:
        print(f"Error converting timestamp : {e}")


def preprocess_df(
    df: pd.DataFrame, verbose: bool = False, phase: Literal["train", "test"] = None
) -> pd.DataFrame:
    """Pipeline wrapper for the whole preprocessing process"""
    df_ = drop_id(df)
    df_ = encode_ram_limit(df_)
    df_ = convert_timestamp(df_)
    df_ = encode_cpu_type(df_)

    if "latency" in df_.columns:
        latency = df_["latency"]
        df_ = df_.drop(columns=["latency"])
    else:
        latency = None

    df_ = df_.astype(float)

    if phase == "train":
        scaler = StandardScaler()
        df_ = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns)

        # Save the scaler to unscale the predictions later on in inference
        joblib.dump(scaler, "../data/Scaler/scaler.pkl")
        print(f"{SUCCESS} Scaler saved successfully")

        if latency is not None:
            latency_scaler = StandardScaler()
            latency = latency_scaler.fit_transform(latency.values.reshape(-1, 1))
            joblib.dump(latency_scaler, "../data/Scaler/latency_scaler.pkl")
            print(f"{SUCCESS} Latency Scaler saved successfully")

    else:
        scaler = joblib.load("../data/Scaler/scaler.pkl")
        df_ = pd.DataFrame(scaler.transform(df_), columns=df_.columns)

        if latency is not None:
            latency_scaler = joblib.load("../data/Scaler/latency_scaler.pkl")
            latency = latency_scaler.transform(latency.values.reshape(-1, 1))

    if latency is not None:
        df_["latency"] = latency.flatten()

    if verbose:
        print(f"{'-' * 120}\n{MAGENTA}Phase: {phase} {RESET}")
        print(
            f"{VERBOSE}[BEFORE]{RESET}\n  - Shape:{df.shape}\n  - Columns:{df.columns.to_list()}"
        )
        print(
            f"{VERBOSE}[AFTER]{RESET}\n  - Shape:{df_.shape}\n  - Columns:{df_.columns.to_list()}\n"
        )
    return df_


def save_data(df: pd.DataFrame, path: str, phase: Literal["train", "test"] = None):
    """Save the dataframe to a csv file"""
    try:
        df.to_csv(path, index=False)
        print(f"{phase} Data saved to {path}")
    except Exception as e:
        print(f"Error saving data : {e}")


if __name__ == "__main__":
    train_df = load_data("../data/training_data.csv")
    test_df = load_data("../data/testing_data.csv")
    processed_train_df = preprocess_df(train_df, verbose=True, phase="train")
    processed_test_df = preprocess_df(test_df, verbose=True, phase="test")
    if SAVE:
        save_data(processed_train_df, "../data/processed_training_data.csv")
        save_data(processed_test_df, "../data/processed_testing_data.csv")
