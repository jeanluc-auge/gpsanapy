# -*- coding: utf-8 -*-

"""
gpsanapy.core.gps_analysis
==================
core gps analytic:
- parse gpx file to html
- format html tags
- parse html with gpxpy library to extract tracks with doppler speed
- import to pandas DataFrame, reindex, resample, plot and save to csv
@ jla, nunu, thelaurent, december 2020
"""

import json
import os
import glob
import datetime
import traceback
from math import pi
from pathlib import Path
from argparse import ArgumentParser
from logging import (
    getLogger,
    basicConfig,
    INFO,
    ERROR,
    DEBUG,
    FileHandler,
    StreamHandler,
)

import gpxpy
import numba
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, element
import matplotlib.pyplot as plt

from utils import (
    log_calls,
    TraceAnalysisException,
    load_config,
    load_results,
    reduce_value_bloc,
    coroutine,
)

TO_KNOT = 1.94384  # * m/s
G = 9.81  #
DEFAULT_REPORT = {"n": 1, "doppler_ratio": None, "sampling_ratio": None, "std": None}
FILE_EXTENSIONS = ['*.sml', '*.gpx']
RECURSIVE_FILE_EXTENSIONS = ['**/*.sml', '**/*.gpx']
DOPPLER_EXCLUSION_LIST = (
    "movescount",
    "waterspeed",
    "suunto",
)  # do not use doppler with these watches
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
LOG_DIR = os.path.join(ROOT_DIR, "csv_results")
# filtering:
MAX_ITER = 10
FILTER_WINDOW = 30
# and using fillna=interpolate

logger = getLogger()
logger.setLevel(INFO)
try:
    log_path = os.path.join(LOG_DIR, "execution.log")
except Exception:
    log_path = os.path.join(ROOT_DIR, "execution.log")
fh = FileHandler(log_path)
fh.setLevel(INFO)
logger.addHandler(fh)
ch = StreamHandler()
ch.setLevel(INFO)
logger.addHandler(ch)


class Trace:
    config_dir = CONFIG_DIR

    def __init__(self, config_file=None):
        if not config_file:
            self.config_file = os.path.join(self.config_dir, "config.yaml")
        else:
            self.config_file = config_file
        self.get_config()
        logger.info(
            f"\nlist of gps functions {self.gps_func_description}\n"
            f"ranking groups vs gps functions: {json.dumps(self.ranking_groups, indent=2)}\n"
            f"ranking functions: {self.ranking_functions}"
        )
        self.mic = self.multi_index_from_config()

    def get_config(self):
        """
        extract info from yaml config files
        Parameters:
            self.config_file
        :return:
            self.rules dict of rules attributs
            self.functions dict of gps function & ranking
            self.directory_paths: dict paths + creation if needed
            self.all_results_path: os.path all results history
        """
        config = load_config(self.config_file)
        self.rules = config.rules
        self.functions = config.functions
        self.directory_paths = config.directory_paths
        # check directory exist or create them:
        for dir_path in self.directory_paths.values():
            if not Path(dir_path).is_dir():
                os.makedirs(dir_path)
        self.all_results_path = os.path.join(
            self.directory_paths.results_dir, "all_results.csv"
        )

    @property
    def gps_func_description(self):
        """
        extract info from yaml config files
        Parameters:
            self.functions
        :return:
            self.gps_func_description
                { fn_description: 'args': {**fn_kwargs} }
        """
        return {
            iteration["description"]: iteration["args"]
            for iterations in self.functions.values()
            for iteration in iterations
        }

    @property
    def ranking_groups(self):
        """
        extract info from yaml config files
        Parameters:
            self.functions
        :return:
            self.ranking_groups
                {ranking_group_name: [fn1_description, ...] }
        """
        ranking_groups = {}
        for gps_func, iterations in self.functions.items():
            for iteration in iterations:
                if iteration.get("ranking_group", "") in ranking_groups:
                    ranking_groups[iteration["ranking_group"]].append(
                        iteration["description"]
                    )
                elif iteration.get("ranking_group", ""):
                    ranking_groups[iteration["ranking_group"]] = [
                        iteration["description"]
                    ]
        return ranking_groups

    @property
    def ranking_functions(self):
        return {
            iteration.description: iteration.args
            for iterations in self.functions.values()
            for iteration in iterations
            if iteration.get("ranking_group", "")
        }

    def reduced_results(
        self,
        by_support="all",
        by_spot="all",
        by_author="all",
        check_config=False,
        all_results=None,
    ):
        params = {}
        if by_support != "all" and by_support:
            params["support"] = by_support
        if by_spot != "all" and by_spot:
            params["spot"] = by_spot
        if by_author != "all" and by_author:
            params["author"] = by_author
        if all_results is None:
            all_results = load_results(self, check_config)
        reduced_results = all_results.copy()
        for param, value in params.items():
            if param in all_results.columns:
                reduced_results = reduced_results[reduced_results[param] == value]
        return reduced_results

    def multi_index_from_config(self):
        # init multiIndex DataFrame for the ranking_results:
        code0 = []
        code1 = []
        code2 = []
        level0 = []
        level1 = list(self.ranking_functions)
        level2 = ["result", "sampling_ratio", "ranking"]
        i = 0
        for k, v in self.ranking_groups.items():
            code0 += len(v) * [i, i, i]
            level0.append(k)
            i += 1
        for i, _ in enumerate(level1):
            code1 += [i, i, i]
            code2 += [0, 1, 2]
        print(code0)
        print(code1)
        print(code2)
        print(level0)
        print(level1)
        print(level2)
        mic = pd.MultiIndex(
            levels=[level0, level1, level2], codes=[code0, code1, code2]
        )
        logger.debug(f"ranking results multi index architecture:" f"{mic}")
        return mic

    # TODO EXCEPTION DECORATOR
    def delete_result(self, filename, file_path):
        parquet_path = os.path.join(
            self.directory_paths.parquet_dir, f"parquet_{filename}"
        )
        all_results = self.reduced_results()
        all_results.drop(index=filename, inplace=True)
        all_results.reset_index(inplace=True)
        all_results.to_csv(self.all_results_path, index=False)
        os.remove(file_path)
        os.remove(parquet_path)

    # TODO EXCEPTION DECORATOR RETURNS None
    def rank_all_results(
        self,
        by_support="all",
        by_spot="all",
        by_author="all",
        check_config=False,
        all_results=None,
        save=False,
    ):
        """
        merge gpx_results of the current filename with all_results history
        and create the ranking_results file based on config yaml ranking groups.
        The ranking may be reduced to certain category with **params
        :param gpx_results: current filename analysis results
        :param params: select ranking by category
            for example: params = dict(spot='g13', support='kitefoil')
        :return: pandas table ranking_results
        """

        # merge DataFrames current gpx_results with all_results history:
        reduced_results = self.reduced_results(
            by_support=by_support,
            by_spot=by_spot,
            by_author=by_author,
            check_config=check_config,
            all_results=all_results,
        )
        if reduced_results.empty:
            return None
        # build ranking_results MultiIndex DataFrame:
        all_results_table = reduced_results[reduced_results.n == 1].pivot_table(
            values=["sampling_ratio", "result"],
            index=["hash"],
            columns=["description"],
            aggfunc=np.mean,
            dropna=False,
        )
        # date_table = self.all_results.groupby('author').date.min()
        ranking_results = pd.DataFrame(index=all_results_table.index, columns=self.mic)
        # rank and fill ranking_results DataFrame:
        ranking = all_results_table.result.rank(
            method="min", ascending=False, na_option="bottom"
        )
        for group, group_func_list in self.ranking_groups.items():
            for description in group_func_list:
                ranking_results.loc[:, (group, description, "ranking")] = ranking[
                    description
                ]
                ranking_results.loc[
                    :, (group, description, "result")
                ] = all_results_table["result"][description]
                ranking_results.loc[
                    :, (group, description, "sampling_ratio")
                ] = all_results_table["sampling_ratio"][description]
        ranking_results.loc[:, "points"] = 0
        for k, v in self.ranking_groups.items():
            ranking_results.loc[:, "points"] += ranking_results.xs(
                (k, "ranking"), level=(0, 2), axis=1
            ).mean(axis=1)
        ranking_results.loc[:, "points"] = ranking_results.loc[:, "points"] / len(
            self.ranking_groups
        )
        ranking_results = ranking_results.sort_values(by=["points"])
        if save:
            ranking_results_path = os.path.join(self.all_results_path, 'ranking_results.csv')
            ranking_results.to_csv(ranking_results_path)
        return ranking_results

class TraceAnalysis:
    # class attributs:
    root_dir = ROOT_DIR  # project root dir (requirements.txt ...)
    config_dir = CONFIG_DIR  # yaml config files location
    analysis_version = 2.2  # track algo improvements
    min_version = 2.2  # min requirement to accept loading from an archive parquet file
    # attributs to save to parquet file:
    #   attributs that cannot be modified
    hard_trace_infos_attr = [
        "parquet_version",
        "analysis_version",
        "creator",
        "trace_sampling",
    ]
    #   user attributs that can be updated
    free_trace_infos_attr = ["author", "spot", "support"]

    def __init__(self, gpx_path, config_file=None, **params):
        """
        :param gpx_path:
        :param config_file:
        :param params:
            author str
            spot str
            support str
            parquet_loading bool : default = config.yaml, overrides config.yaml param
        """
        self.log_info = self.appender("info")
        self.log_warning = self.appender("warning")
        self.gpx_path = gpx_path
        self.filename, self.file_extension = os.path.splitext(gpx_path)
        self.filename = Path(self.filename).stem
        self.trace = Trace(config_file)  # retrive TraceConfig instance client
        for rule, value in self.trace.rules.items():
            setattr(self, rule, value)
        self.sampling = float(self.time_sampling.strip("S"))
        self.parquet_version = (
            None
        )  # version given by the loaded parquet file (if loaded)
        self.params = params
        self.author = params.get("author", self.filename.split("_")[0])
        self.spot = params.get("spot", "")
        self.support = params.get("support", "")
        # **params overrides config.yaml:
        self.parquet_loading = self.params.get("parquet_loading", self.parquet_loading)

    def run(self):
        """
        load file and run the analysis
        :return:
        """
        try:
            self.load_df(self.gpx_path)
            self.set_csv_paths()
            # debug, select a portion of the track:
            # self.df = self.df.loc["2019-03-29 14:10:00+00:00": "2019-03-29 14:47:00+00:00"]
            # generate key time series:
            self.generate_series()
            self.log_trace_infos()
            self.df_result_debug = pd.DataFrame(index=self.tsd.index)
            gpx_results = self.call_gps_func_from_config()
            all_results = self.load_merge_all_results(gpx_results)
            self.ranking_results = self.trace.rank_all_results(all_results=all_results)
            self.save_to_csv()
            return True, f"successfully loaded file {self.filename}"
        except TraceAnalysisException as te:
            logger.error(te)
            return False, [f"TraceAnalysis exception on file {self.filename}\n", f": {te}\n"]
        except Exception as e:
            logger.error(e)
            return False, [
                f"\nan unexpected **{type(e).__name__}** error occured:\n",
                f"{str(e)}\n",
                f" on file {self.filename}\n",
                f"with traceback:\n {traceback.format_exc()}",
            ]

    @coroutine
    def appender(self, level):
        """
        a co routine that logs to logger.{level} and self.log_{level}_list
        :param level: log level
        :return: self.log_{level}_list a list of import logs for the flask api response
        """
        setattr(self, f"log_{level}_list", [])
        to_list = getattr(self, f"log_{level}_list")
        while True:
            log = yield
            to_logger = "\n".join(log)
            getattr(logger, level)(to_logger)
            to_list += log

    def load_df(self, gpx_path):
        self.from_parquet = False
        try:
            if self.load_df_from_parquet():
                self.from_parquet = True
                return
        except Exception as e:
            self.log_info.send(
                [
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    f"failed to load parquet file",
                    f"an unexpected error {e} occured",
                    f"=> resume regular gpx file loading",
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                ]
            )

        self.file_size = Path(gpx_path).stat().st_size / 1e6
        if self.file_size > self.max_file_size:
            raise TraceAnalysisException(
                f"file {gpx_path} size = {self.file_size}Mb > {self.max_file_size}Mb"
            )

        html_soup = self.load_gpx_file_to_html(gpx_path)
        try:
            if not self.load_df_from_sml(html_soup):
                self.creator = html_soup.gpx.get("creator", "unknown")
                tracks = self.format_html_to_gpx(html_soup)
                self.df = self.to_pandas(tracks[0].segments[0])
        except Exception as e:
            raise TraceAnalysisException(
                f"failed to load file {gpx_path} with error {traceback.format_exc()}"
            )
        self.process_df()
        self.resample_df()
        # filter out speed spikes on self.df:
        self.clean_df()
        self.save_df_to_parquet()

    @log_calls()
    def load_df_from_sml(self, html_soup):
        if not self.file_extension == ".sml":
            return False

        self.creator = "suunto sml"

        data_speed = {"speed": [], "time": []}
        data_coor = {"lon": [], "lat": [], "time": []}
        for el in html_soup.findAll("sample"):
            if el.speed and el.utc:
                data_speed["speed"].append(float(el.speed.string))
                data_speed["time"].append(str(el.utc.string))
            elif el.longitude and el.utc:
                data_coor["lon"].append(float(el.longitude.string) * 180 / pi)
                data_coor["lat"].append(float(el.latitude.string) * 180 / pi)
                data_coor["time"].append(str(el.utc.string))
        df_speed = pd.DataFrame(data=data_speed)
        df_speed["time"] = pd.to_datetime(df_speed.time)
        df_speed.set_index("time", inplace=True)
        df_speed = df_speed.resample(self.time_sampling).mean()
        df_speed["has_doppler"] = 0
        df_speed.loc[df_speed.speed.notna(), "has_doppler"] = 1

        # need to resample and inteprolate to obtain speed from distance between 2 points:
        df_coor = pd.DataFrame(data=data_coor)
        df_coor["time"] = pd.to_datetime(df_coor.time)
        df_coor.set_index("time", inplace=True)
        df_coor = df_coor.resample(self.time_sampling).mean().interpolate()
        df_coor["lat-1"] = df_coor["lat"].shift(1)
        df_coor["lon-1"] = df_coor["lon"].shift(1)
        pd_course = lambda x: gpxpy.geo.get_course(
            x["lat"], x["lon"], x["lat-1"], x["lon-1"]
        )
        pd_distance = lambda x: gpxpy.geo.distance(
            x["lat"], x["lon"], None, x["lat-1"], x["lon-1"], None
        )
        df_coor["course"] = df_coor.apply(pd_course, axis=1)
        df_coor["speed_no_doppler"] = df_coor.apply(pd_distance, axis=1) / self.sampling

        # concat the 2 dataframes:
        self.df = pd.concat([df_speed, df_coor], join="outer", axis=1)
        self.df.reset_index(inplace=True)
        self.df.to_csv("test.csv")
        # try:
        #     self.df.index = self.df.index.dt.tz_localize('UTC')
        # except Exception:
        #     pass
        # self.df.index.dt.tz_convert('UTC')
        self.log_info.send(["successful load from sml file {self.filename}"])
        return True

    @log_calls()
    def save_df_to_parquet(self):
        """
        save dataframe to parquet format, before results calculation
        so it can be fastly reloaded and possibly avoid storing large gpx files
        trace infos attributs are also saved in the data frame
        (see class attributs list: cls.trace_infos_attr)
        :return
            save parquet to result_dir/parquet_<filename>
        """
        self.parquet_version = self.analysis_version
        trace_infos_attr = self.free_trace_infos_attr + self.hard_trace_infos_attr
        for attr in trace_infos_attr:
            self.df.loc[self.df.index[0], attr] = getattr(self, attr)
        parquet_path = os.path.join(
            self.trace.directory_paths.parquet_dir, f"parquet_{self.filename}"
        )
        self.df.to_parquet(parquet_path)

    @log_calls()
    def load_df_from_parquet(self):
        """
        load dataframe from parquet file if it exists
        save computation time from gpx & html file reading + filtering
        :return: bool
            True if the dataframe self.df could be loaded from parquet file
        """
        # check files in result_dir:
        filenames = [
            Path(f).stem
            for f in glob.iglob(
                os.path.join(
                    Path(self.trace.directory_paths.parquet_dir).resolve(), "*"
                ),
                recursive=False,
            )
            if (Path(f).stem).startswith("parquet_")
            # if os.path.splitext(f)[1].startswith('parquet_')
        ]
        # search for parquet files matching the filename to analyse:
        parquet_filename = [
            f for f in filenames if self.filename == f.split("parquet_")[1]
        ]
        if not (parquet_filename and self.parquet_loading):
            self.log_info.send(
                [
                    "did not find parquet file to load or config rules prevented parquet loading",
                    "=> resume regular gpx file loading",
                ]
            )
            return False

        # load parquet file:
        parquet_path = os.path.join(
            self.trace.directory_paths.parquet_dir, parquet_filename[0]
        )
        self.log_info.send([f"loading from parquet file {parquet_path}"])
        self.df = pd.read_parquet(parquet_path)

        # check if parquet file is acceptable:
        # (this is about our algo version used to generate the df, it's not about parquet format)
        parquet_version = self.df.loc[self.df.index[0], 'parquet_version']
        if parquet_version < self.min_version:
            self.log_info.send(
                [
                    f"parquet file version {self.parquet_version} is below min version requirement of {self.min_version}",
                    f"=> abort file parquet loading and resume regular gpx file loading",
                ]
            )
            return False

        # load trace attributs:
        for attr in self.hard_trace_infos_attr:
            setattr(self, attr, self.df.loc[self.df.index[0], attr])
        self.df.drop(
            columns=self.hard_trace_infos_attr, inplace=True
        )  # clean df for debug reading
        # free user attributs that can be overriden by the class **params:
        for attr in self.free_trace_infos_attr:
            attr_value = self.params.get(attr, self.df.loc[self.df.index[0], attr])
            setattr(self, attr, attr_value)

        # clean df for debug reading:
        self.df.drop(
            columns=self.free_trace_infos_attr, inplace=True
        )

        return True

    @log_calls()
    def load_gpx_file_to_html(self, gpx_path):
        """
        load gpx file to html processing file format
        :param gpx_path:
        :return:
        """
        try:
            with open(gpx_path, "r") as gpx_file:
                html_soup = BeautifulSoup(gpx_file, "html.parser")
        except Exception as e:
            logger.exception(e)
            raise TraceAnalysisException(
                f"could not open and parse to html the gpx file {gpx_path} with bs4.BeautifulSoup"
            )
        return html_soup

    @log_calls()
    def format_html_to_gpx(self, html_soup):
        """
        remove unwanted tags in html file:
        - <extensions>
        - <gpxdata:speed>
        and replace them with <speed> tag

        :return: gps tracks
        """
        # remove xml description:
        for e in html_soup:
            if isinstance(e, element.ProcessingInstruction):
                e.extract()
        # add <speed> tag:
        for el in html_soup.findAll("gpxdata:speed"):
            el.wrap(html_soup.new_tag("speed"))
        # remove <gpxdata:speed> tag:
        for el in html_soup.findAll("gpxdata:speed"):
            el.unwrap()
        # remove <extensions> tag:
        for el in html_soup.findAll("gpxtpx:trackpointextension"):
            el.unwrap()
        for el in html_soup.findAll("extensions"):
            el.unwrap()
        gpx = gpxpy.parse(str(html_soup), version="1.0")
        tracks = gpx.tracks
        return tracks

    @log_calls(log_args=False, log_result=False)
    def to_pandas(self, raw_data):
        """
        convert gpx track points to pandas DataFrame
        :param raw_data: gpx track points
        :return: pd.DataFrame
        """
        # ******* data frame loading *******
        split_data = [
            {
                "lon": point.longitude,
                "lat": point.latitude,
                "time": point.time,  # datetime.datetime.strptime((str(point.time)).split("+")[0], '%Y-%m-%d %H:%M:%S'),
                "speed": point.speed,
                "doppler_no_doppler": point.speed if point.speed is not None else raw_data.get_speed(i),
                "speed_no_doppler": raw_data.get_speed(i),
                "course": point.course_between(raw_data.points[i - 1] if i > 0 else 0),
                "has_doppler": True if point.speed is not None else False
                # "delta_dist": (
                #     point.distance_3d(raw_data.points[i - 1]) if i > 0 else 0
                # ),
            }
            for i, point in enumerate(raw_data.points)
        ]
        df = pd.DataFrame(split_data)

        # **** data frame doppler checking *****
        # check that there are enough doppler points to interpolate:
        doppler_ratio = 100 - int(100*df.speed.isnull().sum()/len(df.speed))
        if doppler_ratio < 70:
            # not enough doppler speed data:
            # do not interpolate but revert to positional speed when doppler speed is missing:
            logger.info(
                f"insuficient doppler data to be used as raw, ratio is {doppler_ratio}% of doppler\n"
                f"therefore a mix of doppler and positional data will be used to caclulate speed"
            )
            df['speed'] = df["doppler_no_doppler"]

        return df

    @log_calls()
    def process_df(self):
        """
        self.df DataFrame processing:
            - reindexing
            - elapsed time column with cumulated time in s (needed for acceleration)
            - trace base sampling (shortest time between speed samples)
            - convert bool has doppler? column into 0/1 for resampling operations
        + deactivate doppler for Movescount traces
        """

        # reindex on 'time' column for later resample
        try:
            self.df.time = self.df.time.dt.tz_localize("UTC")
        except Exception:
            pass
        self.df.time = self.df.time.dt.tz_convert("UTC")
        self.df = self.df.set_index("time")

        # convert ms-1 to knots:
        self.df.speed = round(self.df.speed * TO_KNOT, 2)
        self.df.speed_no_doppler = round(self.df.speed_no_doppler * TO_KNOT, 2)

        # convert bool to int: needed for rolling window functions
        self.df.loc[self.df.has_doppler == True, "has_doppler"] = 1
        self.df.loc[self.df.has_doppler == False, "has_doppler"] = 0
        self.df["elapsed_time"] = pd.to_timedelta(
            self.df.index - self.df.index[0]
        ).astype("timedelta64[s]")
        sampling = self.df.elapsed_time.diff().mean()
        if sampling < 1:  # round to closest int (np.rint)
            self.trace_sampling = np.rint(sampling*10)/10
        else:  # round down
            self.trace_sampling = np.floor(sampling)

    @log_calls()
    def resample_df(self):
        """
        resample dataframe before filtering
        always resample on cumulated or absolute values (no diff !)
            - resampling to self.time_sampling
            - aggregation (mean/min/max) for under-time_sampling
            - in case of over-time_sampling:
                leave np.nan (raw_df)
                fillna(0) (has doppler?)
                interpolate

        """
        # don't fill nan on gps coordinates:
        tlon = self.df["lon"].resample(self.time_sampling).mean().interpolate()
        tlat = self.df["lat"].resample(self.time_sampling).mean().interpolate()
        # add a new column with total elapsed time in seconds:
        # has_doppler? yes=1 default=0 (np.nan=0), sum = AND (min):
        thd = self.df["has_doppler"].resample(self.time_sampling).min()
        thd = thd.fillna(0).astype(np.int64)
        # speed doppler
        tsd = self.df["speed"].resample(self.time_sampling).mean().interpolate()
        # raw doppler speed = don't fill nan (i.e. no interpolate:
        raw_tsd = self.df["speed"].resample(self.time_sampling).mean()

        # speed no doppler
        ts = (
            self.df["speed_no_doppler"]
            .resample(self.time_sampling)
            .mean()
            .interpolate()
        )
        # raw no doppler speed = don't fill nan (i.e. no interpolate:
        raw_ts = self.df["speed_no_doppler"].resample(self.time_sampling).mean()
        print(ts.shape)
        # course (orientation °) cumulated values => take min of the bin
        tc = self.df["course"].resample(self.time_sampling).min().interpolate()

        df = pd.DataFrame(
            data={
                "lon": tlon,
                "lat": tlat,
                "has_doppler": thd,
                "filtering": np.nan,
                "time_sampling": np.nan,
                "cum_dist": np.nan,
                "delta_dist": np.nan,
                "raw_course": tc,
                "course": tc,
                "course_diff": np.nan,
                "raw_speed": raw_tsd,
                "raw_speed_no_doppler": raw_ts,
                "speed_no_doppler": ts,
                "speed": tsd,
            }
        )
        # generate time_sampling column based on raw_speed:
        df.loc[df.raw_speed.notna(), "time_sampling"] = 1

        df["filtering"] = 0

        self.df = df

    @log_calls()
    def clean_df(self):
        """
        filter self.df on speed_no_doppler and speed fields
        to remove acceleration spikes > max_acceleration
        :return: modify self.df
        """

        # record acceleration (debug):
        self.df["acceleration_doppler_speed"] = self.df.speed.diff() / (
            TO_KNOT * G * self.sampling
        )
        self.df["acceleration_non_doppler_speed"] = self.df.speed_no_doppler.diff() / (
            TO_KNOT * G * self.sampling
        )
        # columns to be filtered out:
        nan_list = [
            "speed",
            "speed_no_doppler",
            "time_sampling",
            "has_doppler",
            "course",
        ]
        erratic_data = True
        iter = 1
        self.filtered_events = 0
        df2 = self.df.copy()

        # limit the # of iterations for speed + avoid infinite loop
        while erratic_data and iter < MAX_ITER:
            err = self.filter_on_field(df2, "speed", "speed_no_doppler")
            self.filtered_events += err
            iter += 1
            erratic_data = err > 0
        self.df.loc[df2[df2.filtering == 1].index, nan_list] = np.nan

        self.df["filtering"] = df2.filtering
        # self.df = self.df[self.df.speed.notna()]

    def filter_on_field(self, df2, *columns):
        """
        filter a given column in self.df
        by checking its acceleration
        and eliminating (filtering) between
        positive (acc>0.4g) and negative (acc < -0.1g) spikes
        param column: df column to process
        :return: Bool assessing if filtering occured
        """
        # 2 filtering algorithms are implemented,
        # small accelerations < aggressive_filtering_th are just filtered out
        # while larger accelerations are analyzed for descending phase and back to normal conditions
        if self.aggressive_filtering:
            aggressive_filtering_th = 1.5 * self.max_acceleration
            filtering_rolling_window = 4
        else:
            aggressive_filtering_th = 1.95 * self.max_acceleration
            filtering_rolling_window = 2
        max_acceleration = self.max_acceleration # needed for numba engine: cannot refer to class attribut
        descend_th = -0.1
        exit_th = 0.005

        @numba.jit(nopython=True)
        def rolling_acceleration_filter(x):
            """
            rolling filter function
            we search for the interval length between
            positive and negative acceleration spikes
            :param
            : x: float64 [60 acceleration samples] array from rolling window
            : descend_th: float triggers descending phase. Exit can only occur after descending phase
            : exit_th: float acceleration value for exit after descending phase
            :return: int # samples count of the interval to filter out
            """
            i = 0
            # searched irrealistic acceleration[0] > max_acceleration spikes
            # we differentiate very high from high acceleration spikes (2 * )
            if x[0] > aggressive_filtering_th:
                exiting = False
                for i, a in enumerate(x):
                    # find spike interval length by searching negative spike end
                    if a < descend_th:
                        # descending phase: continue filtering while a < -0.1
                        exiting = True
                    elif exiting:  # descending phase ends: a>-0.1
                        if a > max_acceleration:
                            exiting = False  # nocheinmal
                        elif a > exit_th:  # wait for reacceleration before exiting
                            break
            return i

        err = 0
        for column in columns:
            column_filtering = f"{column}_filtering"
            column_acceleration = f"{column}_acceleration"
            # calculate g acceleration:
            df2[column_acceleration] = df2[column].diff() / (
                TO_KNOT * G * self.sampling
            )
            acceleration_max = df2[column_acceleration].max()
            if acceleration_max <= self.max_acceleration:  # save execution time
                continue
            # c condition on max_acceleration < acceleration < aggressive_filtering_th
            # condition checked with a rolling max on 2 to 4 samples
            c1 = (
                df2[column_acceleration].rolling(filtering_rolling_window).max()
                <= aggressive_filtering_th
            )
            c2 = (
                df2[column_acceleration].rolling(filtering_rolling_window).max()
                > self.max_acceleration
            )
            c = c1 & c2
            df2.loc[c, column] = np.nan
            df2.loc[c, "filtering"] = 1
            # now filter and apply our rolling acceleration filter
            # on accelerations > aggressive_filtering_th :
            df2[column_filtering] = (
                df2[column_acceleration]
                .rolling(FILTER_WINDOW)
                .apply(rolling_acceleration_filter, engine='numba', raw=True)
                .shift(-FILTER_WINDOW + 1)
            )
            filtering = pd.Series.to_numpy(df2[column_filtering].copy())
            indices = np.argwhere(filtering > 0).flatten()
            err += len(indices)
            for i in indices:
                this_range = df2.iloc[int(i) : int(i + filtering[i]) + 1].index
                df2.loc[this_range, column] = np.nan
                df2.loc[this_range, "filtering"] = 1
            df2.loc[:, column].interpolate(inplace=True)
            self.log_info.send(
                [
                    f"***********************************************\n",
                    f"applied rolling acceleration filter on {column}\n",
                    f"and filtered {len(indices)} events with an acceleration max = {round(acceleration_max,3)}\n",
                ]
            )
        return err

    @log_calls()
    def generate_series(self):
        """
        generate key time series with fillna or interpolate after filtering
        """

        # sunto and apple watches have a False "emulated" doppler that should not be used:
        watch = [x for x in DOPPLER_EXCLUSION_LIST if x in self.creator.lower()]
        if watch and not self.force_doppler_speed:
            self.log_warning.send(
                [
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    f"deactivating doppler for {watch[0]} watches",
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                ]
            )
            self.df["speed"] = self.df.speed_no_doppler
            self.df["has_doppler"] = 0

        # speed doppler
        self.tsd = self.df["speed"].interpolate()
        self.df["speed"] = self.tsd
        self.raw_tsd = self.df["raw_speed"]
        self.raw_ts = self.df["raw_speed_no_doppler"]
        # speed no doppler
        self.ts = self.df["speed_no_doppler"].interpolate()
        self.df["speed_no_doppler"] = self.ts
        # filtering? yes=1 :
        self.tf = self.df["filtering"]
        # time_sampling? yes=1 default=0 (np.nan=0)
        self.tsamp = self.df["time_sampling"]
        self.tsamp = self.tsamp.fillna(0).astype(np.int64)
        self.df["time_sampling"] = self.tsamp
        # has_doppler? yes=1 default=0 (np.nan=0), sum = AND (min):
        self.thd = self.df["has_doppler"]
        self.thd = self.thd.fillna(0).astype(np.int64)
        self.df["has_doppler"] = self.thd
        # interpolate np.nan after filtering
        # (we can because we resampled before filtering, therefore time_sampling is uniform)
        # distance: diff & cumulated calculated from speed and time_sampling:
        self.td = self.tsd * self.sampling / TO_KNOT
        self.tcd = self.td.cumsum()
        self.df["delta_dist"] = self.td
        self.df["cum_dist"] = self.tcd
        # course (orientation °)
        self.df.loc[self.tsd < 5, "course"] = np.nan
        self.tc = self.df.course
        # find a middle between np.nan and interpolate filtered events:
        # fillna = 0 still allows rolling range
        self.tc_diff = self.modulo_diff_ts(self.tc).fillna(0)
        self.df["course"] = self.tc
        self.df["course_diff"] = self.tc_diff
        if self.tsd.max() > self.max_speed:
            raise TraceAnalysisException(
                f"Trace maximum speed after cleaning is = {self.tsd.max()} knots!\n"
                f"abort analysis for speed > {self.max_speed}"
            )

    def log_trace_infos(self):
        """
        compile and log key trace informations
        of the current gpx file under study
        :return: logger
        """
        doppler_ratio = int(100 * len(self.thd[self.thd > 0].dropna()) / len(self.thd))
        sampling_ratio = int(100 * len(self.tsamp[self.tsamp == 1]) / len(self.tsamp))
        if len(self.tsamp[self.tsd > 5]) == 0:
            sampling_ratio_5 = 0
        else:
            sampling_ratio_5 = int(
                100
                * len(self.tsamp[self.tsamp == 1][self.tsd > 5])
                / len(self.tsamp[self.tsd > 5])
            )
        if doppler_ratio < 70:
            self.log_warning.send(
                [
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    f"Doppler speed is not available on all time_sampling points",
                    f"Only {doppler_ratio}% of the points have doppler data",
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                ]
            )
        # if doppler_ratio < 50:
        #     raise TraceAnalysisException(
        #         f"doppler speed is available on only {doppler_ratio}% of data"
        #     )

        if self.trace_sampling != self.sampling:
            self.log_warning.send(
                [
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    f"you are analyzing a trace with a sampling = {self.sampling}S",
                    f"but the trace native sampling = {self.trace_sampling}S",
                    f"over sampling will slow trace analysis for no better precision",
                    f"while under sampling may degrade precision",
                    f"please consider modifying sampling in yaml config",
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                ]
            )
        if self.from_parquet:
            file_info = "loaded from parquet file"
            filtered_events = ""
        else:
            file_info = f"file size is {self.file_size}Mb"
            filtered_events = f"filtered {self.filtered_events} events with acceleration > {self.max_acceleration}g"
        self.log_info.send(
            [
                f"__init__ {self.__class__.__name__} with file {self.gpx_path}",
                f"params {self.params}",
                f"author name: {self.author}",  # trace author: read from gpx file name
                f"spot: {self.spot}",
                f"support: {self.support}",
                file_info,
                f"file loading to pandas DataFrame complete",
                f"creator {self.creator}",  # GPS device type: read from gpx file xml infos field
                f"trace min sampling = {self.trace_sampling}S",
                f"and the trace is analyzed with a sampling = {self.sampling}S",
                f"Trace total distance = {round(self.td.sum() / 1000, 1)} km",
                f"overall doppler_ratio = {doppler_ratio}%",
                f"overall time_sampling ratio = {sampling_ratio}%",
                f"overall time_sampling ratio > 5knots = {sampling_ratio_5}%",
                filtered_events,
                f"now running:    analysis version {self.analysis_version} ",
                f"                parquet version {self.parquet_version}",
            ]
        )

    def modulo_diff_ts(self, ts):
        """
        get diff modulo 360
        :param ts: pd.Series() time serie to process
        :param threshold: threshold of dif event to remove/replace with np.nan
        :return: the filtered time serie in diff()
        """
        ts2 = ts.diff()
        ts_diff = ts.diff()
        # can't interpolate outermost points
        ts2[0] = 0
        ts2[-1] = 0
        ts_diff[0] = 0
        ts_diff[-1] = 0
        ts_diff[ts2 > 180] = ts2 - 360
        ts_diff[ts2 < -180] = ts2 + 360
        return ts_diff

    def diff_clean_ts(self, ts, threshold):
        """
        !! DEPRECATED, replaced by modulo_diff_ts(self) !!
        filter "turn around" events and return the diff of the time serie
        :param ts: pd.Series() time serie to process
        :param threshold: threshold of dif event to remove/replace with np.nan
        :return: the filtered time serie in diff()
        """
        ts2 = ts.diff()
        # can't interpolate outermost points
        ts2[0] = 0
        ts2[-1] = 0
        ts2[abs(ts2) > threshold] = np.nan
        ts2.interpolate(inplace=True)
        return ts2

    @log_calls(log_args=True, log_result=True)
    def v_moy(self, description, v_min=15):
        """
        mean speed above v_min
        :param v_min: float min speed to consider
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """
        result = round(self.tsd[self.tsd > v_min].mean(), 1)
        results = [{"result": result, "description": description, **DEFAULT_REPORT}]
        return results

    @log_calls(log_args=True, log_result=True)
    def planning_ratio(self, description, v_min=15, distance=True):
        """
        % (ratio) of distance or time spent while v > v_min
        :param
            description: str
            v_min: float knots min speed to consider
            distance: bool True=ratio on distance, False=ratio on time
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """
        if distance:
            result = int(100 * self.td[self.tsd > v_min].sum() / self.td.sum())
        else:
            result = int(100 * len(self.tsd[self.tsd > v_min]) / len(self.tsd))
        results = [{"result": result, "description": description, **DEFAULT_REPORT}]
        return results

    @log_calls(log_args=True, log_result=True)
    def planning_distance(self, description, v_min=15):
        """
        total distance covered while v > v_min
        :param vmin: float knots min speed to consider
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """
        result = round(int(self.td[self.tsd > v_min].agg(sum)) / 1000, 1)
        results = [{"result": result, "description": description, **DEFAULT_REPORT}]
        return results

    @log_calls(log_args=True, log_result=True)
    def speed_jibe(self, description, n=5):
        """
        calculate the best jibe min speeds
        :param n: int number of records
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """
        PARTIAL_COURSE = 75
        FULL_COURSE = 135
        MIN_JIBE_SPEED = 11
        speed_window = int(np.ceil(20 / self.sampling))
        speed_shift = int(np.ceil(13 / self.sampling))
        full_course_window = int(np.ceil(16 / self.sampling))  # 16s
        partial_course_window = int(np.ceil(full_course_window / 3))  # 5s
        rolling_extension = int(np.ceil(full_course_window / 6))
        tc = self.tc_diff.copy()
        ts = self.tsd.copy()

        # =====================================================================
        # CONDITION 0 = min speed > MIN_JIBE_SPEED in the speed_window around the jibe (center)
        # i.e. speed > 9knots in a 20s window
        # remove low speed periods (too many noise in course orientation) on speed_window:
        tc[self.tsd < MIN_JIBE_SPEED] = np.nan
        # consider a speed_exclusion zone around min speed:
        ts[ts < MIN_JIBE_SPEED] = np.nan
        # ts[self.tsd.rolling(speed_shift, center=True).min() < MIN_JIBE_SPEED] = np.nan
        # arbitrarily remove start and end of session:
        tc.iloc[0:30] = np.nan
        tc.iloc[-30:-1] = np.nan
        # =====================================================================
        # CONDITION 1 = cumulated course > HALF_JIBE_COURSE in the partial_course_window1 around jibe (center)
        # do not include partial rolling data (no min_periods)
        # i.e. cumulated course > 70° in a 5s window
        #   => rolling(partial_course_window, center=True).sum()
        # + check this condition in the last partial_course_window2 samples
        #   => rolling(partial_course_window_2).max()
        # find condition 1 on 5 samples rolling window:
        j1 = (
            abs(tc.rolling(partial_course_window, center=True).sum())
            .rolling(rolling_extension)
            .max()
        )
        cj1 = j1 > PARTIAL_COURSE
        # =====================================================================
        # CONDITION 2 = cumulated course > FULL_COURSE in the full_course_window around jibe (center)
        # do not include partial rolling data
        # i.e. cumulated course > 150° in 15s window
        #   => rolling(full_course_window, center=True).sum()
        # find condition2 on 15 samples rolling window:
        j2 = (
            abs(tc.rolling(full_course_window, center=True).sum())
            .rolling(rolling_extension)
            .max()
        )
        cj2 = j2 > FULL_COURSE
        # =====================================================================
        j3 = abs(tc.rolling(rolling_extension, center=True).sum())
        # TODO ? condition on speed dip: search for inflexion points in speed
        # acceleration1 = round(ts.ewm(2).mean().shift(-1), 2)
        # acceleration2 = acceleration1.diff().diff().ewm(3).mean().shift(-2)
        # acceleration3 = round(
        #     ts.diff().rolling(4, center=True).mean().diff().ewm(4).mean().shift(-2), 2
        # )
        # jibe_speed3 = ts.copy()
        # jibe_speed3[acceleration2<0.1] = np.nan
        # jibe_speed3 = jibe_speed3.rolling(full_course_window, center=True, min_periods=1).min()
        # =====================================================================
        # APPLY CONDITION 1  & 2
        self.jibe_range = cj1 & cj2  # save it to plot for debug
        # search for min speed, including partial rolling data
        jibe_speed12 = ts.rolling(speed_window, min_periods=1).min().shift(-speed_shift)
        # jibe_speed12 = ts.rolling(speed_window, center=True, min_periods=1).min()
        jibe_speed12[~(self.jibe_range)] = np.nan
        # =====================================================================
        # identify center of jibe = highest instantaneous course

        course_max = j3.copy()
        course_max[~self.jibe_range] = np.nan
        reduced_course_max = reduce_value_bloc(course_max, roll_func="max")
        jibe_speed123 = jibe_speed12.copy()
        jibe_speed123[course_max - reduced_course_max < -0.01] = np.nan
        jibe_speed123[course_max < PARTIAL_COURSE / 2] = np.nan

        # ====== debug starts =====================
        self.df["cj1"] = j1
        self.df["cj2"] = j2
        self.df["cj3"] = j3
        self.df["speed cj1+cj2+cj3"] = jibe_speed123.copy()
        self.df["speed cj1+cj2"] = jibe_speed12.copy()
        self.df["reduced_course_max"] = reduced_course_max
        self.df["ts"] = ts
        # self.raw_df["speed 3"] = jibe_speed3
        # # ====== debug ends =====================

        # generate a list of all jibes min speed on a 20 samples window for conditions 1 & 2:
        jibe_speed = jibe_speed123[self.jibe_range]
        results = []
        if len(jibe_speed) == 0:
            logger.warning(f"could not find any valid jibe")
            # abort: could not find any valid jibe
            return [
                {
                    "result": None,
                    "description": description,
                    **DEFAULT_REPORT,
                    "n": i + 1,
                }
                for i in range(n)
            ]
        for i in range(1, n + 1):
            # find the highest speed jibe index and speed and define a [-11s, +11s] centered window
            range_begin = jibe_speed.idxmax() - datetime.timedelta(seconds=9)
            range_end = jibe_speed.idxmax() + datetime.timedelta(seconds=14)
            if range_end is not np.nan:
                result = round(jibe_speed.dropna().max(), 2)

                # remove this speed range to find others:
                jibe_speed[range_begin:range_end] = 0
                confidence_report = self.append_result_debug(
                    item_range=self.tsd[range_begin:range_end].index,
                    item_description=description,
                    item_iter=i,
                )
            else:
                confidence_report = DEFAULT_REPORT
                result = None
            results.append(
                {
                    "description": description,
                    "result": result,
                    **confidence_report,
                    "n": i,
                }
            )
        nvjibe_speed_results = "\n".join(
            [f"{result['description']}: {result['result']}" for result in results]
        )
        logger.info(
            f"\n===============================\n"
            f"Best jibe min speed x {n} \n"
            f"{nvjibe_speed_results}"
            f"\n===============================\n"
        )
        return results

    @log_calls(log_args=True, log_result=True)
    def speed_dist(self, description, dist=500, n=5):
        """
        calculate Vmax n x V[distance]
        :param dist: float distance to consider for speed mean
        :param n: int number of vmax to record
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """

        # find a starting point from the max distance in a sample (vmax)
        samples_count = int(dist / self.td.max()) + 7
        logger.info(f"starting search with {samples_count} samples")
        k = 1
        results = []
        tsd = self.tsd.copy()
        td = self.td.copy()
        while k < n + 1:
            iter = 0
            while iter < 50:  # avoid infinite loop
                max_rolling_distance_1 = td.rolling(samples_count - 1).sum().max()
                max_rolling_distance_2 = td.rolling(samples_count).sum().max()
                if max_rolling_distance_1 > dist:
                    samples_count -= 1
                elif max_rolling_distance_2 < dist:
                    samples_count += 1
                else:
                    break
                iter += 1

            distance = max_rolling_distance_2
            rolling_speed = tsd.rolling(samples_count).mean()
            rolling_distance = td.rolling(samples_count).sum()
            result = round(rolling_speed.max(), 2)
            range_end = rolling_speed.idxmax()
            if range_end is not np.nan:
                range_begin = range_end - datetime.timedelta(
                    seconds=int(samples_count * self.sampling) - 1
                )
                tsd.loc[range_begin:range_end] = 0
                td.loc[range_begin:range_end] = 0
                logger.info(
                    f"found {samples_count} samples for n={k}:\n"
                    f"max rolling({samples_count}) speed = {result}\n"
                    f"max rolling({samples_count}) distance = {distance}\n"
                )
                confidence_report = self.append_result_debug(
                    item_range=self.tsd[range_begin:range_end].index,
                    item_description=description,
                    item_iter=k,
                )
            else:
                confidence_report = DEFAULT_REPORT
                result = None
            results.append(
                {
                    "description": description,
                    "result": result,
                    **confidence_report,
                    "n": k,
                }
            )
            k += 1

        nvdist_speed_results = "\n".join(
            [f"{result['description']}: {result['result']}" for result in results]
        )
        logger.info(
            f"\n===============================\n"
            f"Best vmax {n} x {dist}m\n"
            f"{nvdist_speed_results}"
            f"\n===============================\n"
        )
        return results

    @log_calls(log_args=True, log_result=True)
    def deprecated_speed_dist(self, description, dist=500, n=5):
        """
        calculate Vmax n x V[distance]
        :param dist: float distance to consider for speed mean
        :param n: int number of vmax to record
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """

        def rolling_dist_count(delta_dist):
            """
            rolling distance filter
            count the number of samples needed to reach a given distance
            :param dist: float distance to reach
            :return: int  # samples to reach distance dist
            """
            delta_dist = delta_dist[::-1]
            cum_dist = 0
            for i, d in enumerate(delta_dist):
                cum_dist += d
                if cum_dist > dist:
                    break
            return i + 1

        min_speed = 7  # knots min expected speed for max window size
        max_interval = dist / min_speed
        max_samples = int(max_interval / self.sampling)

        td = self.td.rolling(max_samples).apply(rolling_dist_count)
        nd = pd.Series.to_numpy(td)
        if np.isnan(nd).all():
            return [{"result": 0, "description": description, **DEFAULT_REPORT}]
        min_samples = int(np.nanmin(nd))
        threshold = min(min_samples + 10, max_samples)
        logger.info(
            f"\nsearching {n} x v{dist} speed\n"
            f"over a window of {max_samples} samples\n"
            f"and found a min of {min_samples*self.sampling} seconds\n"
            f"to cover {dist}m"
        )
        logger.info(
            f"checking result:\n"
            f"max rolling({min_samples}) speed = {self.tsd.rolling(min_samples).mean().max()}\n"
            f"max rolling({min_samples}) distance = {self.td.rolling(min_samples).sum().max()}\n"
        )

        indices = np.argwhere(nd <= threshold).flatten()
        indices_range = [set(range(int(i - nd[i]), int(i))) for i in indices]
        # create a list of tupple (speed Vmax_dist, speed_indice_range)
        speed_list = [(self.tsd[range_i].mean(), range_i) for range_i in indices_range]
        speed_list.sort(key=lambda tup: tup[0], reverse=True)
        k = 1
        total_range = set([])
        results = []
        for speed, speed_range in speed_list:
            if not (speed_range & total_range):
                result = round(speed, 1)
                confidence_report = self.append_result_debug(
                    item_range=self.tsd[speed_range].index,
                    item_description=description,
                    item_iter=k,
                )
                results.append(
                    {
                        "description": description,
                        "result": result,
                        "n": k,
                        **confidence_report,
                    }
                )
                total_range = total_range | speed_range
                k += 1
            if k > n:
                break

        nvdist_speed_results = "\n".join(
            [f"{result['description']}: {result['result']}" for result in results]
        )
        logger.info(
            f"\n===============================\n"
            f"Best vmax {n} x {dist}m\n"
            f"{nvdist_speed_results}"
            f"\n===============================\n"
        )
        return results

    @log_calls(log_args=True, log_result=True)
    def speed_xs(self, description, s=10, n=10):
        """
        calculate Vmax: n * x seconds
        :param xs: int time interval in seconds
        :param n: number of records
        :return: results dict
            description
            result
            n
            doppler_ratio
            sampling_ratio
            std
        """
        # to str:
        xs = f"{s}S"

        # select n best Vmax:
        nxs_list = []
        results = []
        tsd = self.tsd.copy()
        exclusion_time = datetime.timedelta(seconds=10)
        for i in range(1, n + 1):
            # calculate s seconds Vmax on all data:
            ts = tsd.rolling(xs).mean()
            range_end = ts.idxmax()
            range_begin = range_end - datetime.timedelta(seconds=s - 1)
            result = round(ts.max(), 2)
            # remove this speed range to find others:
            tsd[range_begin - exclusion_time : range_end + exclusion_time] = 0
            # generate debug report:
            confidence_report = self.append_result_debug(
                item_range=self.tsd[range_begin:range_end].index,
                item_description=description,
                item_iter=i,
            )
            results.append(
                {
                    "description": description,
                    "result": result,
                    "n": i,
                    **confidence_report,
                }
            )

        nxs_speed_results = "\n".join(
            [f"{result['description']}: {result['result']}" for result in results]
        )
        logger.info(
            f"\n===============================\n"
            f"Best vmax {n} x {s} seconds\n"
            f"{nxs_speed_results}"
            f"\n===============================\n"
        )
        return results

    def append_result_debug(self, item_range, item_description, item_iter):
        """
        add result to the result_debug report
        log all data related to the analysis during its period of time
        :param item_range: DatetimeIndex period of time of the analysis
        :param item_description: str description of the analysis to record
        :param item_iter: int n x () iteration of the analysis (i.e. n x Vmax10s)
        :return:
            pd.DataFrame self.df.result_debug
            confidence_report {doppler_ratio, sampling_ratio, std dev}
        """
        self.df_result_debug.loc[item_range, "has_doppler?"] = self.thd[item_range]
        self.df_result_debug.loc[item_range, "filtering?"] = self.tf[item_range]
        self.df_result_debug.loc[item_range, "time_sampling?"] = self.tsamp[item_range]
        self.df_result_debug.loc[item_range, "speed"] = self.tsd[item_range]
        self.df_result_debug.loc[item_range, "raw_speed"] = self.raw_tsd[item_range]
        self.df_result_debug.loc[item_range, "speed_no_doppler"] = self.ts[item_range]
        self.df_result_debug.loc[item_range, "course"] = self.tc[item_range]
        self.df_result_debug.loc[item_range, "course_diff_cleaned"] = self.tc_diff[
            item_range
        ]
        self.df_result_debug.loc[item_range, "dist"] = self.td[item_range]
        self.df_result_debug.loc[item_range, item_description] = item_iter
        # generate report:
        doppler_ratio = int(
            100
            * len(self.thd[item_range][self.thd > 0].dropna())
            / len(self.thd[item_range])
        )
        sampling_ratio = int(
            100
            * len(self.tsamp[item_range][self.tsamp == 1])
            / len(self.tsamp[item_range])
        )
        std = round(self.tsd[item_range].std(), 2)
        confidence_report = dict(
            doppler_ratio=doppler_ratio, sampling_ratio=sampling_ratio, std=std
        )
        logger.debug(
            f"\nconfidence_report on {item_description} in n={item_iter}:\n"
            f"{confidence_report}"
        )
        return confidence_report

    @log_calls(log_args=False, log_result=True)
    def call_gps_func_from_config(self):
        """
        generate the session performance summary
        load config.yaml file with instructions
        about gps analysis functions to call with args
        and record their result in self.result DataFrame
        :return: pandas df
            author
            creator
            date
            description
            result
            doppler_ratio
            sampling_ratio
            std
        """
        results = []
        # iterate over the config and call the referenced functions:

        for gps_func, iterations in self.trace.functions.items():
            # the same gps_func key cannot be repeated in the yaml description,
            # so we use an iterations list,
            # in order to call several times the same function with different args if needed:
            for iteration in iterations:
                results += getattr(self, gps_func)(
                    description=iteration["description"], **iteration["args"]
                )

        # update results with gpx file creator and author and convert to df:
        date = str(self.df.index[0].date())
        if self.spot:
            hash = f"{self.author}-{date}-{self.spot}"
        else:
            hash = f"{self.author}-{date}"
        data = [
            dict(
                hash=hash,
                filename=self.filename,
                creator=self.creator,
                author=self.author,
                support=self.support,
                spot=self.spot,
                date=date,
                **result,
            )
            for result in results
        ]
        gpx_results = pd.DataFrame(data=data)
        gpx_results = gpx_results.set_index("filename")
        self.gpx_results = gpx_results
        return gpx_results

    def load_merge_all_results(self, gpx_results):
        """
        load all_results history swap file and merge with current analysis results
        :param gpx_results:
            pandas df indexed by filename of current analysis results
        :return:
            self.all results pandas df of results merged with history
        """
        # merge DataFrames current gpx_results with all_results history
        all_results = load_results(self.trace, check_config=True)
        if all_results is None:
            all_results = gpx_results
        elif self.filename in all_results.index:
            all_results.loc[self.filename, :] = gpx_results
        else:  # merge
            all_results = pd.concat([all_results, gpx_results])
        logger.debug(
            f"\nloaded all results history and merged with {self.filename} results:\n"
            f"{all_results.head(30)}\n"
        )
        self.all_results = all_results
        return all_results

    def set_csv_paths(self):
        results_dir = self.trace.directory_paths.results_dir
        # debug file with the full DataFrame (erased at each run):
        debug_filename = "debug.csv"
        # debug file reduced to the main results timeframe (new for different authors):
        result_debug_filename = f"{self.filename}_result_debug.csv"
        # result file of the current run (new for different authors):
        result_filename = f"{self.filename}_result.csv"
        # all time history results by user names (updated after each run):
        all_results_filename = "all_results.csv"
        # all time history results table with ranking (re-created at each run):
        if self.support:
            ranking_results_filename = f"ranking_results_{self.support}.csv"
        else:
            ranking_results_filename = f"ranking_results.csv"

        self.debug_path = os.path.join(results_dir, debug_filename)
        self.result_debug_path = os.path.join(results_dir, result_debug_filename)
        self.results_path = os.path.join(results_dir, result_filename)
        self.all_results_path = self.trace.all_results_path
        self.ranking_results_path = os.path.join(results_dir, ranking_results_filename)

    @log_calls()
    def save_to_csv(self):
        """
        save to csv file the simulation results and infos (debug)
        :return: 5 csv files
            - self.raw_df (debug.csv) with the full DataFrame
            - filename_result_debug.csv with the runs details of each result
            - filename_result.csv result summary
            - all_results.csv history swap file
            - ranking_results.csv history compilation/presentation file
        """
        # *********** self.raw_df: debug.csv ***********
        self.df.to_csv(self.debug_path)
        # *********** self.df_result_debug: filename_result_debug.csv ***********
        result_debug = self.df_result_debug[self.df_result_debug.speed.notna()]
        result_debug.to_csv(self.result_debug_path)
        # *********** gpx_results: filename_result.csv ***********
        self.gpx_results.to_csv(self.results_path, index=False)
        # ****** self.all_results _history swap file_: all_results.csv *******
        self.all_results = self.all_results[
            self.all_results.creator.notna()
        ].reset_index()
        self.all_results.to_csv(self.all_results_path, index=False)
        # **** self.ranking_results history output file ranking_results.csv
        self.ranking_results.to_csv(self.ranking_results_path)

    def log_computation_time(self):
        fn_execution_time = {
            fn: f"executed in {round(time, 2)} s and {round(100 * time / self.fn_execution_time['total_time'],1)}% of total time"
            for fn, time in self.fn_execution_time.items()
            if fn != "total_time"
        }
        logger.info(
            f"\n===============================================\n"
            f"total computation time is {round(self.fn_execution_time['total_time'],2)} s\n"
            f"with the following repartition:\n"
            f"{json.dumps(fn_execution_time, indent=2)}"
            f"===============================================\n"
        )

    @log_calls()
    def plot_speed(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        dfs = pd.DataFrame(index=self.tsd.index)
        dfs["raw_speed"] = self.raw_tsd
        dfs["speed"] = self.tsd
        dfs["speed_no_doppler"] = self.ts
        # dfs["delta_doppler"] = self.ts - self.tsd
        try:
            dfs.plot(ax=ax1)
        except Exception:
            logger.error(f"cannot plot speed")
        try:
            tjr = self.tc_diff[self.jibe_range]
            data = {
                "diff_course_speed>10": self.tc_diff,  # [self.tsd > 7],
                "cum_course_speed>10": self.tc,  # [self.tsd > 7],
                "distance": self.td,
                "jibe_range": tjr,
            }
            dfc = pd.DataFrame(index=self.tsd.index, data=data)
            dfc.plot(ax=ax2)
        except Exception:
            logger.error(f"cannot plot distance and course")
            # stupid error of delta index I cant't be bothered with
        plt.show()


def process_args(args):
    filenames = []
    f = args.gpx_filename
    rd = args.recursive_read_directory
    d = args.read_directory

    if f:
        filenames = f
    elif d:
        for ext in FILE_EXTENSIONS:
            filenames.extend(glob.glob(os.path.join(Path(d).resolve(), ext), recursive=False))
    elif rd:
        for ext in RECURSIVE_FILE_EXTENSIONS:
            filenames.extend(glob.glob(os.path.join(Path(rd).resolve(), ext), recursive=True))

    logger.info(f"\nthe following files will be processed:\n" f"{filenames}")
    return filenames


def crunch_data():
    """
    crunch all results data
    :return: hostory plots analysis
    """
    from utils import process_config_plot

    trace = Trace()
    config_plot_file = os.path.join(TraceAnalysis.config_dir, "config_plot.yaml")
    process_config_plot(trace.reduced_results(), config_plot_file)


parser = ArgumentParser()
parser.add_argument("-f", "--gpx_filename", nargs="+", type=Path)
parser.add_argument(
    "-rd", "--recursive_read_directory", nargs="?", type=str, default=""
)
parser.add_argument("-d", "--read_directory", nargs="?", type=str, default="")
parser.add_argument("-p", "--plot", action="count", default=0)
parser.add_argument("-c", "--crunch_data", action="count", default=0)
parser.add_argument("-data", "--params_data", nargs="?", type=json.loads, default={})
# !! mind the quotes with data json loads !! use:
#       -data '{"author": "jla", ...}'

# parser.add_argument(
#     "-v",
#     "--verbose",
#     action="count",
#     default=0,
#     help="increases verbosity for each occurence",
# )

if __name__ == "__main__":
    args = parser.parse_args()
    # basicConfig(level={0: INFO, 1: DEBUG}.get(args.verbose, INFO))
    config_filename = os.path.join(
        TraceAnalysis.config_dir, "config.yaml"
    )  # config of gps functions to call
    filenames = process_args(args)
    error_dict = {}
    all_results = None
    for filename in filenames:
        params = args.params_data
        gpsana_client = TraceAnalysis(filename, config_filename, **params)
        status, error = gpsana_client.run()
        if not status:
            error_dict[filename] = error
        gpsana_client.log_computation_time()
        if args.plot > 0:
            gpsana_client.plot_speed()

    if args.crunch_data > 0:
        crunch_data()
        plt.show()

    for f, e in error_dict.items():
        logger.error(
            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"could not process file {f}: {e}\n"
            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
