import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class DataframeUtils:

    @staticmethod
    def convert_df_from_pandas_to_r(pdf: pd.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.py2rpy(pdf)

    @staticmethod
    def convert_df_from_r_to_pandas(obj_r):
        with localconverter(ro.default_converter + pandas2ri.converter):
            return pandas2ri.rpy2py(obj_r)
