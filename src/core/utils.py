import inspect
import itertools
import functools
import json
import os
import yaml
import logging

logger = logging.getLogger()

def log_calls(log_args=False, log_result=False):

    def outer_wrap(fn):
        """
        class method decorator: intensive & pretty logging interest
        log the method name and args (excluding self)

        args
            fn: the method to be decorated
        return:
            the decorated method

        """

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            # checking func signature:
            sig_func = inspect.signature(fn).parameters
            # positional args, not including self:
            posargs_list = [k for k, v in sig_func.items() if '**' not in str(v)][1:]
            # bind them to values in their positional order, even if no value is passed (zip_longest):
            fill_value = '!no arg value found!'  # to be replaced by default values or TypeError: missing a required argument
            posargs = dict(itertools.zip_longest(posargs_list, args, fillvalue=fill_value))
            # complete with kwargs:
            posargs.update(**kwargs)
            # complete with default args that were not passed:
            defargs = {
                k: v.default
                for k, v in sig_func.items()
                if ('=' in str(v) and posargs[k] == fill_value)
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
            result =  fn(self, *args, **kwargs)
            # logging results:
            if log_result:
                logger.info(
                    f"\nFUNCTION {fn.__name__} ENDS\n"
                    f"with result:\n"
                    f"{result}\n"
                    f"======================================================"
                )
            return result

        return wrapper

    return outer_wrap

class TraceAnalysisException(Exception):
    def __init__(self, body):
        self.body = body

    def __str__(self):
        return (
            f"\n=======================================\n"
            f"GPS analysis critical error:\n"
            f"{self.body}\n"
            f"======================================="
        )

def load_config(config_file=None):
    """Load config files

    yaml|yml suffixed file path :return: the config object

    Args:
        config_file: str path to file
    Return:
        config dict object
    """
    # support cascade of function calls with defautl value:
    if config_file is None:
        config_file = "config.yaml"
    logger.info(f"loading yaml config from file {config_file}")
    root, ext = os.path.splitext(config_file)

    try:
        with open(config_file) as file:
            if ext in [".yaml", ".yml"]:
                config = yaml.safe_load(file.read())
            else:
                logger.error(
                    f"Param's file {config_file} extension is not valid"
                )
                return
    except Exception as e:
        logger.error(
            f"Failed to load workflow from file {config_file}: {e}"
        )
        return
    print(config)
    return config

