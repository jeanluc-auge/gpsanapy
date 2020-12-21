import inspect
import itertools
import functools
import json
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
                f"\n\n======================================================\n"
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
                    f"======================================================\n"
                )
            return result

        return wrapper

    return outer_wrap


def reindex(df, time_col):
    """
        set index of data frame to colon time_col
        Args:
            df: Pandas DataFrame to process
            time_co: str colon name with time

    """

    # set TIMEFRAME colum as index:
    df2 = df.set_index(time_col)
    return df2

def resample(df, sampling_tag=None):
    """
    resample to sampling_tag
    !! df needs to be aggregated
    Args:
        sampling_tag: time sample (1d, 1h, 1min)
        data_rage: [0-1] percentage of data to select

    """
    if not sampling_tag:  # allow function cascade of default param
        sampling_tag = "2S"
    df2 = df.resample(sampling_tag).speed.mean()
    logger.info(f"serie resampled to {sampling_tag} frequency")
    logger.debug(
        f"dataframe head before resample:\n {df.head(15)}"
    )  # there are duplicates
    logger.debug(f"dataframe head after resample:\n {df2.head(15)}")
    return df2