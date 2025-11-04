import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

class DatasetUtils:

    @staticmethod
    def load_dataset():
        """ Load dataset and return features and class names """
        wine = load_wine(as_frame=True)
        X, y = wine.data, wine.target
        names = list(wine.target_names)
        df = X.copy()
        df["target"] = y.map(lambda i: names[i]).astype("category")
        return df, names

    @staticmethod
    def split_df(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size, random_state=seed, stratify=y)
        train_df = X_train.copy(); train_df["target"] = y_train
        test_df = X_test.copy(); test_df["target"] = y_test
        return train_df, test_df