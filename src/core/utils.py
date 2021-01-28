import inspect
import itertools
import functools
import json
import os
import yaml
import logging
import pandas as pd
import numpy as np
from time import time

logger = logging.getLogger()


def log_calls(log_args=False, log_result=False):
    def outer_wrap(fn):
        """
        class method decorator: intensive & pretty logging interest
        log
            method name
            execution time
            optional: args (excluding self) and results
        args
            fn: the method to be decorated
        return:
            the decorated method
            self.fn_execution_time dict

        """

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            # checking func signature:
            sig_func = inspect.signature(fn).parameters
            # positional args, not including self:
            posargs_list = [k for k, v in sig_func.items() if "**" not in str(v)][1:]
            # bind them to values in their positional order, even if no value is passed (zip_longest):
            fill_value = (
                "!no arg value found!"
            )  # to be replaced by default values or TypeError: missing a required argument
            posargs = dict(
                itertools.zip_longest(posargs_list, args, fillvalue=fill_value)
            )
            # complete with kwargs:
            posargs.update(**kwargs)
            # complete with default args that were not passed:
            defargs = {
                k: v.default
                for k, v in sig_func.items()
                if ("=" in str(v) and posargs[k] == fill_value)
            }
            posargs.update(**defargs)
            func_arg_dict = posargs
            logger.info(
                f"\n======================================================\n"
                f"CALLING FUNCTION {fn.__name__}\n"
            )
            if log_args:
                logger.info(f"with args: {func_arg_dict}\n")
            # calling function
            start_time = time()
            result = fn(self, *args, **kwargs)
            end_time = time()
            # logging results:
            execution_time = end_time - start_time

            logger.info(
                f"\nFUNCTION {fn.__name__} ENDS in {round(execution_time*1000, 0)}ms\n"
            )
            if log_result:
                logger.info(
                    f"with result:\n"
                    f"{result}\n"
                    f"======================================================"
                )
            fn_name = func_arg_dict.get('description', fn.__name__)
            if not hasattr(self, 'fn_execution_time'): # init
                self.fn_execution_time = {'total_time': 0}
            # don't log time of call_gps_func_from_config because it calls every other functions
            if fn_name != 'call_gps_func_from_config':
                self.fn_execution_time[fn_name] = execution_time
                self.fn_execution_time['total_time'] += execution_time

            return result

        return wrapper

    return outer_wrap


class TraceAnalysisException(Exception):
    def __init__(self, body):
        self.body = body

    def __str__(self):
        return (
            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"GPS analysis critical error:\n"
            f"{self.body}\n"
            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )


def reduce_value_bloc(ts, window=3):
    """
    takes pd Series with islands of values between np.nan
    and return a pd Serie with islands of values reduced to the min of the island
    :param df:
    :return: Pandas Time Serie with islands of values between np.nan reduced to the min of the bloc
    """
    i = 1
    ts0 = ts.copy()
    while i < 5:
        # iterate backward:
        ts1 = ts0.rolling(window).min().shift(-window+1).fillna(ts0)
        #iterate forward:
        ts0 = ts1.rolling(window).min().fillna(ts1)
        i += 1
    return ts0

def load_config(config_filename=None):
    """Load config files

    yaml|yml suffixed file path :return: the config object

    Args:
        config_filename: str path to file
    Return:
        config dict object
    """
    # support cascade of function calls with defautl value:
    if config_filename is None:
        config_filename = "config.yaml"
    logger.info(f"loading yaml config from file {config_filename}")
    root, ext = os.path.splitext(config_filename)

    try:
        with open(config_filename) as file:
            if ext in [".yaml", ".yml"]:
                config = yaml.safe_load(file.read())
            else:
                logger.error(f"Param's file {config_filename} extension is not valid")
                return
    except Exception as e:
        logger.error(f"Failed to load workflow from file {config_filename}: {e}")
        return
    return config


def load_results(gps_func_dict, all_results_filename):
    """
    open csv all_results_filename with history of other previous sessions or other riders
    and load it into a pd.DataFrame
        - if the history all_results_filename is missing, a new history file is created with the same name
        - if the yaml config does not match the opened history file,
            a new history file is created with the same name
            and the older saved under "all_results_old.csv"
    :param gps_func_list: list func to calls from yaml config file
    :param all_results_filename: str name of the history file
    :return: pd.DataFrame all_results from csv all_results_filename
    """
    old_all_results_filename = "all_results_old.csv"
    # open filename if it exists:
    try:
        all_results = pd.read_csv(all_results_filename)
        all_results = all_results.set_index("author")
    except Exception as e:
        logger.warning(
            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"{str(e)}\n"
            f"{all_results_filename} is missing\n"
            f"=> a new file will be created"
            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
        return None

    # check the loaded results are compatible with the current yaml config:
    config_error = ''
    all_results_gps_func_list = list(all_results.groupby('description').mean().index)
    if set(all_results_gps_func_list) ^ set(list(gps_func_dict)):
        config_error = f"all_results.csv:\n {all_results_gps_func_list}"
    else:
        for description in gps_func_dict:
            if all_results.groupby('description').max().loc[description, 'n'] != gps_func_dict[description].get('n',1):
                config_error = f"{(all_results.groupby('description').max().loc[description, 'n'])}"
    if config_error:
        logger.error(
            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"config in yaml file does not match the config used for all_results.csv\n"
            f"yaml func list:{gps_func_dict}\n"
            f"vs {config_error}\n"
            f"a new all_results.csv file will be created with no ranking history\n"
            f"ranking history in current all_results.csv will be saved as all_results_old.csv\n"
            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
        all_results = all_results.reset_index()
        all_results.to_csv(old_all_results_filename)
        all_results = None
    return all_results
