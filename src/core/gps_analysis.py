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
from pathlib import Path
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG
import gpxpy
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, element
import matplotlib.pyplot as plt


from utils import log_calls, TraceAnalysisException, load_config, load_results

logger = getLogger()

MAX_SPEED = 45 # knots
MAX_ACCELERATION = 0.5 # g or +5m/s/s or +10 knots/s, negative acc is not limited ;)
MAX_FILE_SIZE = 10e6
DEFAULT_REPORT = {"n": 1, "doppler_ratio": None, "sampling_ratio": None, "std": None}

class TraceAnalysis:
    def __init__(self, gpx_path, config_file='config.yaml', sampling="1S"):
        self.version = "10th January 2021"
        self.sampling = sampling
        self.gpx_path = gpx_path
        self.filename = Path(self.gpx_path).stem
        self.set_csv_paths()
        self.config = load_config(config_file)
        self.process_config()
        self.df = self.load_df(gpx_path)
        self.process_df()
        author = self.filename.split("_")[0]
        self.author = f"{author}_{str(self.df.index[0].date())}"
        # debug, select a portion of the trac:
        #self.df = self.df.loc["2019-04-02 16:34:00+00:00": "2019-04-02 16:36:00+00:00"]
        # original copy that will not be modified: for reference & debug:
        self.raw_df = self.df.copy()
        # filter out speed spikes on self.df:
        self.clean_df()
        # generate key time series:
        self.generate_series()
        self.log_trace_infos()
        self.df_result_debug = pd.DataFrame(index=self.tsd.index)

    def load_df(self, gpx_path):
        self.file_size = Path(gpx_path).stat().st_size
        if  self.file_size > MAX_FILE_SIZE:
            raise TraceAnalysisException(
                f"file {gpx_path} size = {self.file_size/1e6}Mb > {MAX_FILE_SIZE/1e6}Mb"
            )
        html_soup = self.load_gpx_file_to_html(gpx_path)
        if not html_soup.gpx:
            raise TraceAnalysisException("the loaded gpx file is empty")
        self.creator = html_soup.gpx.get("creator", "unknown")
        tracks = self.format_html_to_gpx(html_soup)
        df = self.to_pandas(tracks[0].segments[0])
        return df

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
                "time": point.time,
                "speed": (point.speed if point.speed else raw_data.get_speed(i)),
                "speed_no_doppler": raw_data.get_speed(i),
                "course": point.course_between(raw_data.points[i - 1] if i > 0 else 0),
                "has_doppler": bool(point.speed),
                "delta_dist": (
                    point.distance_2d(raw_data.points[i - 1]) if i > 0 else 0
                ),
            }
            for i, point in enumerate(raw_data.points)
        ]
        df = pd.DataFrame(split_data)
        # **** data frame doppler checking *****
        if not df.has_doppler.all():
            ts = pd.Series(data=0, index=df.index)
            ts[df.has_doppler == True] = 1
            doppler_ratio = int(100 * sum(ts) / len(ts))
            if doppler_ratio < 70:
                logger.warning(
                    f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"Doppler speed is not available on all sampling points\n"
                    f"Only {doppler_ratio}% of the points have doppler data\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )
            # if doppler_ratio < 50:
            #     raise TraceAnalysisException(
            #         f"doppler speed is available on only {doppler_ratio}% of data"
            #     )
        return df

    def process_df(self):
        """
        self.df DataFrame processing
        """

        # reindex on 'time' column for later resample
        self.df = self.df.set_index("time")

        # convert ms-1 to knots:
        self.df.speed = round(self.df.speed * 1.94384, 2)
        self.df.speed_no_doppler = round(self.df.speed_no_doppler * 1.94384, 2)

        # add a new column with total elapsed time in seconds:
        self.df["elapsed_time"] = pd.to_timedelta(
            self.df.index - self.df.index[0]
        ).astype("timedelta64[s]")

        # add a cumulated distance column (cannot resample on diff!!)
        self.df["cum_dist"] = self.df.delta_dist.cumsum()

        # convert bool to int: needed for rolling window functions
        self.df.loc[self.df.has_doppler == True, "has_doppler"] = 1
        self.df.loc[self.df.has_doppler == False, "has_doppler"] = 0

        # sunto watches have a False "emulated" doppler that should not be used:
        if "movescount" in self.creator.lower():
            logger.warning(
                f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"deactivating doppler for Movescount watches\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )
            self.df["speed"] = self.df.speed_no_doppler
            self.df["has_doppler"] = 0

    @log_calls()
    def clean_df(self):
        """
        filter self.df on speed_no_doppler and speed fields
        to remove acceleration spikes > 0.5g
        :return: modify self.df
        """
        # record acceleration (debug):
        self.raw_df['acceleration'] = self.raw_df.speed.diff() / (
            9.81 * self.raw_df.elapsed_time.diff()
        )
        erratic_data = True
        iter = 1
        self.filtered_events = 0
        df2 = self.df.copy()
        df2['filtering'] = 0
        # limit the # of iterations for speed + avoid infinite loop
        while erratic_data and iter < 5:
            #err1 = self.filter_on_field(df2, "speed_no_doppler")
            err = self.filter_on_field(df2, "speed")
            self.filtered_events += err
            erratic_data = err>0
            iter += 1
        self.df.loc[df2[df2.filtering==1].index] = np.nan
        self.raw_df["filtering"] = df2.filtering
        self.df = self.df[self.df.speed.notna()]

    def filter_on_field(self, df2, column):
        """
        filter a given column in self.df
        by checking its acceleration
        and eliminating (filtering) between
        positive (acc>0.5g) and negative (acc < -0.1g) spikes
        param column: df column to process
        :return: Bool assessing if filtering occured
        """

        def rolling_acceleration_filter(x):
            """
            rolling filter function
            we search for the interval length between
            positive and negative acceleration spikes
            :param x: float64 [60 acceleration samples] array from rolling window
            :return: int # samples count of the interval to filter out
            """
            i = 0
            # searched irrealisitc >0.5g acceleration spikes
            if x[0] > MAX_ACCELERATION:
                exiting = False
                for i, a in enumerate(x):
                    # find spike interval length by searching negative spike end
                    if a < -0.1:
                        exiting = True
                    elif exiting and a != np.nan:
                        if a < MAX_ACCELERATION:
                            break
                        else:
                            exiting = False
            return i

        column_filtering = f"{column}_filtering"

        # calculate g acceleration:
        df2['acceleration'] = df2[column].diff() / (
            9.81 * df2.elapsed_time.diff()
        )
        # apply our rolling acceleration filter:
        df2[column_filtering] = (
            df2.acceleration
            .rolling(20)
            .apply(rolling_acceleration_filter)
            .shift(-19)
        )
        filtering = pd.Series.to_numpy(df2[column_filtering].copy())
        indices = np.argwhere(filtering > 0).flatten()
        for i in indices:
            this_range = df2.iloc[int(i): int(i + filtering[i]) + 1].index
            df2.loc[this_range, column] = np.nan
            df2.loc[this_range, 'filtering'] = 1

        # 2 possibilities: interpolate or fill forward the last notna value
        # if we don't do any of the 2, we cannot iterate,
        # we chose to interpolate because the forward fill
        # can make believe (false) high acceleration spikes after long np.nan filtering
        # it's a rectangular spike vs triangular spike for the interpolate, hence smoother:
        df2[column].interpolate(inplace=True)
        #df2[column].fillna(method = 'ffill', inplace=True)
        return len(indices)

    def generate_series(self):
        """
        generate key time series with
            - resampling to self.sampling
            - aggregation (mean/min/max) for up-sampling
            - optional interpolate in case of down-sampling (else np.nan)
        """

        # speed doppler
        self.tsd = self.df["speed"].resample(self.sampling).mean().interpolate()
        self.raw_tsd = self.raw_df["speed"].resample(self.sampling).mean()
        # speed no doppler
        self.ts = (
            self.df["speed_no_doppler"].resample(self.sampling).mean().interpolate()
        )
        # filtering? yes=1 default=0 (np.nan=0), sum = OR (max):
        self.tf = self.raw_df["filtering"].resample(self.sampling).max()
        self.tf = self.tf.fillna(0).astype(np.int64)
        # no sampling? yes=1 default=1 (np.nan=1), sum = OR
        self.tno_samp = self.raw_df["filtering"].resample(self.sampling).max()
        self.tno_samp = self.tno_samp.fillna(1).astype(np.int64)
        # has_doppler? yes=1 default=0 (np.nan=0), sum = AND (min):
        self.thd = self.df["has_doppler"].resample(self.sampling).min()
        self.thd = self.thd.fillna(0).astype(np.int64)
        # distance: resample on cumulated values => take min of the bin and restore diff:
        self.tcd = self.df["cum_dist"].resample(self.sampling).min().interpolate()
        self.td = self.tcd.diff()
        # course (orientation °) cumulated values => take min of the bin
        self.tc = self.df["course"].resample(self.sampling).min().interpolate()
        self.tc_diff = self.diff_clean_ts(self.tc, 300)

        if self.tsd.max() > MAX_SPEED:
            raise TraceAnalysisException(
                f"Trace maximum speed after cleaning is = {self.tsd.max()} knots!\n"
                f"abort analysis for speed > {MAX_SPEED}"
            )

    def log_trace_infos(self):
        doppler_ratio = int(100 * len(self.thd[self.thd > 0].dropna()) / len(self.thd))
        sampling_ratio = int(
            100 * len(self.tno_samp[self.tno_samp == 0].dropna()) / len(self.tno_samp)
        )
        if len(self.tno_samp[self.tsd>5]) == 0:
            sampling_ratio_5 = 0
        else:
            sampling_ratio_5 = int(
                100 * len(self.tno_samp[self.tno_samp == 0][self.tsd>5].dropna()) / len(self.tno_samp[self.tsd>5])
            )
        logger.info(
            f"\n==========================================================================\n"
            f"==========================================================================\n"
            f"__init__ {self.__class__.__name__} with file {self.gpx_path}\n"
            f"author name: {self.author}\n"  # trace author: read from gpx file name
            f"file size is {self.file_size/1e6}Mb\n"
            f"file loading to pandas DataFrame complete\n"
            f"creator {self.creator}\n"  # GPS device type: read from gpx file xml infos field
            f"total distance {round(self.td.sum()/1000,1)} km"
            f"\noverall doppler_ratio = {doppler_ratio}%\n"
            f"overall sampling ratio = {sampling_ratio}%\n"
            f"overall sampling ratio > 5knots = {sampling_ratio_5}%\n"
            f"filtered {self.filtered_events} events with acceleration > 0.5g\n"
            f"now running version {self.version}\n"
            f"==========================================================================\n"
            f"==========================================================================\n"
        )

    def diff_clean_ts(self, ts, threshold):
        """
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
        :return: float mean speed of the session above v_min
        """
        result = round(self.tsd[self.tsd > v_min].mean(), 1)
        results = [{"result": result, "description": description, **DEFAULT_REPORT}]
        return results

    @log_calls(log_args=True, log_result=True)
    def planning_ratio(self, description, v_min=15, distance=True):
        """
        ratio of distance or time spent while v > v_min
        :param
            description: str
            v_min: float knots min speed to consider
            distance: bool True=ratio on distance, False=ratio on time
        :return: the % of time spent over v_min
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
        :return: the total distance spent over v_min
        """
        result = round(int(self.td[self.tsd > v_min].agg(sum)) / 1000, 1)
        results = [{"result": result, "description": description, **DEFAULT_REPORT}]
        return results

    @log_calls(log_args=True, log_result=True)
    def speed_jibe(self, description, n=5):
        """
        calculate the best jibe min speeds
        :param n: int number of records
        :return: list of n * vmin jibe speeds
        """
        sampling = int(self.sampling.strip("S"))
        HALF_JIBE_COURSE = 70
        FULL_JIBE_COURSE = 130
        MIN_JIBE_SPEED = 11
        speed_window = int(20 / sampling)
        course_window = int(15 / sampling)
        partial_course_window = int(course_window / 3)

        tc = self.tc_diff.copy()
        # remove low speed periods (too many noise in course orientation):
        tc[self.tsd.rolling(speed_window, center=True).min() < MIN_JIBE_SPEED] = np.nan
        tc.iloc[0:30] = np.nan
        tc.iloc[-30:-1] = np.nan
        # find consition 1 on 5 samples rolling window:
        cj1 = (
            abs(tc.rolling(partial_course_window, center=True).sum()) > HALF_JIBE_COURSE
        )
        # find condition2 on 15 samples rolling window:
        cj2 = abs(tc.rolling(course_window, center=True).sum()) > FULL_JIBE_COURSE

        # # ====== debug starts =====================
        # df = pd.DataFrame(index=tc.index)
        # df['c'] = self.tc
        # df['tc'] = tc
        # df['tc5']= tc.rolling(5, center=True).sum()
        # df['tc15'] = tc.rolling(15, center=True).sum()
        # df['tsd'] = self.tsd
        # df['tsd20'] = self.tsd.rolling(20, center=True).min()
        # df['r']=self.tsd.rolling(20, center=True).min()[cj1 & cj2]
        # df.to_csv('debug_jibe.csv')
        # # ====== debug ends =====================

        # generate a list of all jibes min speed on a 20 samples window for conditions 1 & 2:
        jibe_speed = self.tsd.rolling(speed_window, center=True).min()[cj1 & cj2]
        results = []
        if len(jibe_speed) == 0:
            # abort: could not find any valid jibe
            return [{"description": description, "result": None, **DEFAULT_REPORT}]
        for i in range(1, n + 1):
            # find the highest speed jibe index and speed and define a [-11s, +11s] centered window
            range_begin = jibe_speed.idxmax() - datetime.timedelta(seconds=11)
            range_end = jibe_speed.idxmax() + datetime.timedelta(seconds=11)
            result = round(jibe_speed.dropna().max(), 1)

            # remove this speed range to find others:
            jibe_speed[range_begin:range_end] = 0
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
        return results

    @log_calls(log_args=True, log_result=True)
    def deprecated_speed_jibe(self, description, n=5):
        """
        !! DEPRECATED: DO NOT USE !!
        calculate the best jibe min speeds
        :param n: int number of records
        :return: list of n * vmin jibe speeds
        """
        HALF_JIBE_COURSE = 70
        FULL_JIBE_COURSE = 130
        MIN_JIBE_SPEED = 11

        # filter "crazy Yvan" events on course (orientation),
        # that is: remove all 360° glitchs:
        tc = self.diff_clean_ts(self.tc, 300)
        # sum course over 5S and 20S windows:
        tj1 = tc.rolling("5S").sum()
        tj2 = tc.rolling("15S").sum().shift(-5)
        # record min speeds over a 10S window:
        tsd = self.tsd.rolling("10S").min()

        nsj1 = pd.Series.to_numpy(tj1)
        nsj2 = pd.Series.to_numpy(tj2)
        nsd = pd.Series.to_numpy(tsd)
        # find indices where course changed by more than 70° in 5S
        indices_j1 = np.argwhere(abs(nsj1) > HALF_JIBE_COURSE).flatten()
        # # find indices where course changed by more than 130° in 20S
        indices_j2 = np.argwhere(abs(nsj2) > FULL_JIBE_COURSE).flatten()
        # # find min speeds > 10 knots:
        # indices_d = np.argwhere(abs(nsd) > MIN_JIBE_SPEED).flatten()
        # # merge: new set with elements common to j and d:
        # indices = list(set(indices_j1) & set(indices_j2) & set(indices_d))
        # record a 20S range around these indices:
        indices_range = [
            set(range(int(i - 10), int(i + 10)))
            for i in indices_j1
            if i > 1 and i < len(tsd) - 11
        ]
        jibe_list = [
            (
                self.tsd[range_i].min(),  # jibe min speed in the 20S range
                range_i,  # jibe range
            )
            for range_i in indices_range
        ]
        # reverse sort on jibe speed:
        jibe_list.sort(key=lambda tup: tup[0], reverse=True)

        # iterate to find n x best jibes v
        k = 1
        total_range = set([])
        results = []

        for jibe_speed, jibe_range in jibe_list:
            # remove overlapping ranges (jibe already recorded):
            if not (jibe_range & total_range):
                confidence_report = self.append_result_debug(
                    item_range=self.tsd[jibe_range].index,
                    item_description=description,
                    item_iter=k,
                )
                results.append(
                    {
                        "description": description,
                        "result": jibe_speed,
                        "n": k,
                        **confidence_report,
                    }
                )
                total_range = total_range | jibe_range  # append jibe range
                k += 1
            if k > n:
                break
        return results

    @log_calls(log_args=True, log_result=True)
    def speed_dist(self, description, dist=500, n=5):
        """
        calculate Vmax n x V[distance]
        :param dist: float distance to consider for speed mean
        :param n: int number of vmax to record
        :return: vmax mean over distance dist
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

        sampling = int(self.sampling.strip("S"))
        min_speed = 7  # knots min expected speed for max window size
        max_interval = dist / min_speed
        max_samples = int(max_interval / sampling)

        td = self.td.rolling(max_samples).apply(rolling_dist_count)
        nd = pd.Series.to_numpy(td)
        if np.isnan(nd).all():
            return [{"result": 0, "description": description, **DEFAULT_REPORT}]
        min_samples = int(np.nanmin(nd))
        threshold = min(min_samples + 10, max_samples)
        logger.info(
            f"\nsearching {n} x v{dist} speed\n"
            f"over a window of {max_samples} samples\n"
            f"and found a min of {min_samples*sampling} seconds\n"
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

        # ts.index.get_loc(ts2.idxmin('index'))
        # tsd.plot()
        # plt.show()
        return results

    @log_calls(log_args=True, log_result=False)
    def speed_xs(self, description, s=10, n=10):
        """
        calculate Vmax: n * x seconds
        :param xs: int time interval in seconds
        :param n: number of records
        :return: list of n * vs
        """
        # to str:
        xs = f"{s}S"

        # select n best Vmax:
        nxs_list = []
        results = []
        tsd = self.tsd.copy()
        for i in range(1, n + 1):
            # calculate s seconds Vmax on all data:
            ts = tsd.rolling(xs).mean()
            range_end = ts.idxmax()
            range_begin = range_end - datetime.timedelta(seconds=s - 1)
            result = round(ts.max(), 1)
            # remove this speed range to find others:
            tsd[range_begin:range_end] = 0
            # generate debug report
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
        self.df_result_debug.loc[item_range, "no_sampling?"] = self.tno_samp[item_range]
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
            * len(self.tno_samp[item_range][self.tno_samp == 0].dropna())
            / len(self.tno_samp[item_range])
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

    @log_calls()
    def process_config(self):
        ranking_groups = {}

        for gps_func, iterations in self.config.items():
            for iteration in iterations:
                if iteration["ranking_group"] in ranking_groups:
                    ranking_groups[iteration["ranking_group"]].append(
                        iteration["description"]
                    )
                else:
                    ranking_groups[iteration["ranking_group"]] = [
                        iteration["description"]
                    ]
        self.gps_func_description = [
            gps_func for v in ranking_groups.values() for gps_func in v
        ]
        self.ranking_groups = ranking_groups
        logger.info(
            f"\nlist of gps functions {self.gps_func_description}\n"
            f"ranking groups vs gps functions: {self.ranking_groups}\n"
        )

    def multi_index_from_config(self):
        # init multiIndex DataFrame for the ranking_results:
        code0 = []
        code1 = []
        code2 = []
        level0 = []
        level1 = self.gps_func_description
        level2 = ["result", "sampling_ratio", "ranking"]
        i = 0
        for k, v in self.ranking_groups.items():
            code0 += len(v) * [i, i, i]
            level0.append(k)
            i += 1
        for i, _ in enumerate(level1):
            code1 += [i, i, i]
            code2 += [0, 1, 2]
        mic = pd.MultiIndex(
            levels=[level0, level1, level2], codes=[code0, code1, code2]
        )
        logger.debug(f"ranking results multi index architecture:" f"{mic}")
        return mic

    @log_calls(log_args=True, log_result=True)
    def call_gps_func_from_config(self):
        """
        generate the session performance summary
        load config.yaml file with instructions
        about gps analysis functions to call with args
        and record their result in self.result DataFrame
        :return: pd.DataFrame() self_result
        """
        results = []
        # iterate over the config and call the referenced functions:

        for gps_func, iterations in self.config.items():
            # the same gps_func key cannot be repeated in the yaml description,
            # so we use an iterations list,
            # in order to call several times the same function with different args if needed:
            for iteration in iterations:
                results += getattr(self, gps_func)(
                    description=iteration["description"], **iteration["args"]
                )

        # update results with gpx file creator and author and convert to df:
        data = [
            dict(creator=self.creator, author=self.author, date=str(self.df.index[0].date()), **result)
            for result in results
        ]
        gpx_results = pd.DataFrame(data=data)
        gpx_results = gpx_results.set_index("author")
        # ordered (wrto ranking) list of gps_func to call:
        return gpx_results

    def merge_all_results(self, gpx_results):
        # merge DataFrames current gpx_results with all_results history
        all_results = load_results(self.gps_func_description, self.all_results_path)
        if all_results is None:
            all_results = gpx_results
        elif self.author in all_results.index:
            all_results.loc[self.author] = gpx_results
        else:  # merge
            all_results = pd.concat([all_results, gpx_results])
        logger.debug(
            f"\nloaded all results history and merged with {self.author} results:\n"
            f"{all_results.head(30)}\n"
        )
        return all_results

    @log_calls(log_args=False, log_result=True)
    def rank_all_results(self, gpx_results):
        # merge DataFrames current gpx_results with all_results history:
        self.all_results = self.merge_all_results(gpx_results)
        # build ranking_results MultiIndex DataFrame:
        all_results_table = self.all_results[self.all_results.n == 1].pivot_table(
            values=["sampling_ratio", "result"],
            index=["author"],
            columns=["description"],
            aggfunc=np.mean,
            dropna=False,
        )
        #date_table = self.all_results.groupby('author').date.min()
        ranking_results = pd.DataFrame(index=all_results_table.index, columns=self.multi_index_from_config())
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
        self.ranking_results = ranking_results
        return ranking_results

    @log_calls()
    def plot_speed(self):
        dfs = pd.DataFrame(index=self.tsd.index)
        dfs["raw_speed"] = self.raw_tsd
        dfs["speed"] = self.tsd
        dfs["speed_no_doppler"] = self.ts
        # dfs["delta_doppler"] = self.ts - self.tsd
        dfs.plot()
        plt.show()

    def set_csv_paths(self):
        result_directory = os.path.join(
            os.path.dirname(__file__),
            f"../../csv_results"
        )
        # debug file with the full DataFrame (erased at each run):
        debug_filename = "debug.csv"
        # debug file reduced to the main results timeframe (new for different authors):
        result_debug_filename = f"{self.filename}_result_debug.csv"
        # result file of the current run (new for different authors):
        result_filename = f"{self.filename}_result.csv"
        # all time history results by user names (updated after each run):
        all_results_filename = "all_results.csv"
        # all time history results table with ranking (re-created at each run):
        ranking_results_filename = ("ranking_results.csv")

        self.debug_path = os.path.join(result_directory, debug_filename)
        self.result_debug_path = os.path.join(result_directory, result_debug_filename)
        self.results_path = os.path.join(result_directory, result_filename)
        self.all_results_path = os.path.join(result_directory, all_results_filename)
        self.ranking_results_path = os.path.join(result_directory, ranking_results_filename)

    @log_calls()
    def save_to_csv(self, gpx_results):
        """
        save to csv file the simulation results and infos (debug)
        :return: 3 csv files
            - debug.csv with the full DataFrame after filtering
            - result_debug.csv with the runs details of each result
            - result.csv result summary
        """
        self.raw_df.to_csv(self.debug_path)
        result_debug = self.df_result_debug[self.df_result_debug.speed.notna()]
        result_debug.to_csv(self.result_debug_path)
        gpx_results.to_csv(self.results_path, index=False)
        if hasattr(self, 'all_results'):
            self.all_results = self.all_results[self.all_results.creator.notna()].reset_index()
            self.all_results.to_csv(self.all_results_path, index=False)
            self.ranking_results.to_csv(self.ranking_results_path)


def process_args(args):
    f = args.gpx_filename
    d = args.read_directory

    if f:
        gpx_filenames = f
    if d:
        gpx_filenames = [
            f for f in glob.iglob(
                os.path.join(Path(d).resolve(), '**/*.gpx'), recursive=True
            )
        ]
    logger.info(f"\nthe following gpx files will be processed:\n" f"{gpx_filenames}")
    return gpx_filenames


parser = ArgumentParser()
parser.add_argument("-f", "--gpx_filename", nargs="+", type=Path)
parser.add_argument("-rd", "--read_directory", nargs="?", type=str, default="")
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="increases verbosity for each occurence",
)

if __name__ == "__main__":
    args = parser.parse_args()
    basicConfig(level={0: INFO, 1: DEBUG}.get(args.verbose, INFO))
    config_filename = "config.yaml"  # config of gps functions to call
    gpx_filenames = process_args(args)
    error_dict = {}
    for gpx_filename in gpx_filenames:
        try:
            gpx_jla = TraceAnalysis(gpx_filename, config_filename)
            gpx_results = gpx_jla.call_gps_func_from_config()
            gpx_jla.rank_all_results(gpx_results)
            #gpx_jla.plot_speed()
            gpx_jla.save_to_csv(gpx_results)
        except TraceAnalysisException as te:
            error_dict[gpx_filename] = str(te)
            logger.error(te)
        except Exception as e:
            error_dict[gpx_filename] = (
                f"\nan unexpected **{type(e).__name__}** error occured:\n"
                f"{str(e)}\n"
                f"with traceback:\n {traceback.format_exc()}"
            )
            logger.error(e)
    for f, e in error_dict.items():
        logger.error(
            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"could not process file {f}: {e}\n"
            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )