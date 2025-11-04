from __future__ import annotations

import os
import sys
import logging
from c5_train_predict import C5TrainPredict
from dataset_utils import DatasetUtils

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Python: %s", sys.version)
logger.info("R: %s", os.environ.get("R_HOME"))

if __name__ == "__main__":
    # load
    df, class_names = DatasetUtils.load_dataset()
    train_df, test_df = DatasetUtils.split_df(df, test_size=0.25, seed=42)

    # train
    fit, helpers = C5TrainPredict.train(train_df)

    # predict
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].astype(str).tolist()
    y_pred = C5TrainPredict.predict(fit, helpers, X_test)
    y_pred_converted = [class_names[i - 1] for i in y_pred]

    # evaluate
    logger.info("Results")
    output = C5TrainPredict.evaluate_report(y_test, y_pred_converted, class_names)
    logger.info(output)
