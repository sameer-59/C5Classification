import pandas as pd
from dataclasses import dataclass
from rpy2.robjects.packages import importr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataframe_utils import DataframeUtils


@dataclass
class C5Params:
    trials: int = 10
    rules: bool = False
    winnow: bool = True


class C5TrainPredict:

    @staticmethod
    def train(train_df: pd.DataFrame):
        c5 = importr("C50")
        c5_params = C5Params(trials=10, rules=False, winnow=True)
        r_train = DataframeUtils.convert_df_from_pandas_to_r(train_df)
        target_pos = train_df.columns.get_loc("target") + 1
        x_train = r_train.rx(True, -target_pos)
        y_train = r_train.rx2("target")

        fit = c5.C5_0(x=x_train, y=y_train,
                      trials=c5_params.trials,
                      rules=c5_params.rules,
                      winnow=c5_params.winnow)
        return fit, {"c5": c5}

    @staticmethod
    def predict(fit, helpers, X_test):
        c5 = helpers["c5"]
        r_testX = DataframeUtils.convert_df_from_pandas_to_r(X_test)
        pred_factor = c5.predict_C5_0(fit, r_testX)
        return list(pred_factor)

    @staticmethod
    def evaluate_report(y_test, y_pred, class_names) -> str:
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4, target_names=class_names)
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_names],
                             columns=[f"pred_{c}" for c in class_names])
        return (
            f"Accuracy: {acc:.4f}\n\n"
            f"Classification report:\n{report}\n"
            f"Confusion matrix:\n{cm_df}\n"
        )
