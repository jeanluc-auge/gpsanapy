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
import datetime
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG
import gpxpy
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, element
import matplotlib.pyplot as plt


from utils import log_calls, reindex, resample, TraceAnalysisException, load_config

logger = getLogger()

DEFAULT_RESULTS = {
    'n': 1,
    'doppler_ratio': None,
    'sampling_ratio': None,
    'std': None,
}

class TraceAnalysis:
    def __init__(self, gpx_path, sampling="1S"):
        logger.info(f"init {self.__class__.__name__} with file {gpx_path}")
        self.sampling = sampling
        self.gpx_path = gpx_path
        self.df = self.load_df(gpx_path)
        # original copy that will not be modified: for reference & debug:
        self.raw_df = self.df.copy()
        begin = "2019-04-02 16:34:00+00:00"
        end = "2019-04-02 16:36:00+00:00"
        # self.select_period(begin, end)
        # filter out speed spikes on self.df:
        self.clean_df()
        # generate key time series with fixed sampling and interpolation:
        self.tsd = resample(self.df, sampling, "speed")
        self.raw_tsd = resample(self.raw_df, sampling, "speed", fillna=False)
        self.thd = resample(self.df, sampling, "has_doppler", fillna=False)
        self.ts = resample(self.df, sampling, "speed_no_doppler")
        self.td = resample(self.df, sampling, "cum_dist")
        self.td = self.td.diff()
        self.tc = resample(self.df, sampling, "course")
        self.df_result_debug = pd.DataFrame(index=self.tsd.index)

    def load_df(self, gpx_path):
        html_soup = self.load_gpx_file_to_html(gpx_path)
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
                f"could not open and parse to html the gpx file {gpx_path}"
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

    @log_calls(log_args=False, log_result=True)
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
        # reindex:
        df = reindex(df, "time")
        # *************************************
        # ******* data frame processing *******
        # convert ms-1 to knots:
        df.speed = round(df.speed * 1.94384, 2)
        df.speed_no_doppler = round(df.speed_no_doppler * 1.94384, 2)
        # add a new column with total elapsed time in seconds:
        df["elapsed_time"] = pd.to_timedelta(df.index - df.index[0]).astype(
            "timedelta64[s]"
        )
        # add a cumulated distance column (cannot resample on diff!!)
        df["cum_dist"] = df.delta_dist.cumsum()
        # **************************************
        # **** data frame doppler checking *****
        if not df.has_doppler.all():
            ts = pd.Series(data=0, index=df.index)
            ts[df.has_doppler == True] = 1
            doppler_ratio = int(100 * sum(ts) / len(ts))
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
        # **************************************
        return df

    def filter_on_field(self, column):
        """
        filter a given column in self.df
        by checking its acceleration
        and eliminating (filtering) between
        positive (acc>0.3g) and negative (acc < -0.1g) spikes
        param column: df column to process
        :return: Bool assessing if filtering occured
        """

        def filter(x):
            """
            rolling filter function
            we search for the interval length between
            positive and negative acceleration spikes
            :param x: float64 [60 acceleration samples] array from rolling window
            :return: int # samples count of the interval to filter out
            """
            i = 0
            # searched irrealisitc >0.3g acceleration spikes
            if x[0] > 0.4:
                exiting = False
                for i, a in enumerate(x):
                    # find spike interval length by searching negative spike end
                    if a < -0.1:
                        exiting = True
                    elif exiting:
                        break
            return i

        acceleration = f"{column}_acceleration"
        filtering = f"{column}_filtering"
        self.df[acceleration] = self.df[column].diff() / (
            9.81 * self.df.elapsed_time.diff()
        )
        self.df[filtering] = self.df[acceleration].rolling(30).apply(filter).shift(-29)
        filtering = pd.Series.to_numpy(self.df[filtering])
        indices = np.argwhere(filtering > 0).flatten()
        for i in indices:
            self.df.iloc[int(i) : int(i + filtering[i])] = np.nan
        self.df = self.df[self.df.speed.notna()]
        return len(indices) > 0

    @log_calls()
    def clean_df(self):
        """
        filter self.df on speed_no_doppler and speed fields
        to remove acceleration spikes > 0.3g
        :return: modify self.df
        """
        erratic_data = True
        iter = 1
        while erratic_data and iter < 10:
            #erratic_data = self.filter_on_field("speed_no_doppler")
            erratic_data = erratic_data or self.filter_on_field("speed")
            iter += 1

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
        result = round(self.tsd[self.tsd > v_min].mean(), 2)
        results = [
            {
                'result': result,
                "description": description,
                **DEFAULT_RESULTS
            }
        ]
        return results

    @log_calls(log_args=True, log_result=True)
    def planning_ratio(self, description, v_min=15):
        """
        ratio of time spent while v > v_min
        :param v_min: float knots min speed to consider
        :return: the % of time spent over v_min
        """
        result = int(100 * len(self.tsd[self.tsd > v_min]) / len(self.tsd))
        results = [
            {
                'result': result,
                "description": description,
                **DEFAULT_RESULTS
            }
        ]
        return results

    @log_calls(log_args=True, log_result=True)
    def planning_distance(self, description, v_min=15):
        """
        total distance covered while v > v_min
        :param vmin: float knots min speed to consider
        :return: the total distance spent over v_min
        """
        result = int(self.td[self.tsd > v_min].agg(sum)) / 1000
        results = [
            {
                'result': result,
                "description": description,
                **DEFAULT_RESULTS
            }
        ]
        return results

    @log_calls(log_args=True, log_result=True)
    def speed_jibe(self, description, n=5):
        """
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
        tj2 = tc.rolling("20S").sum()
        # record min speeds over a 10S window:
        tsd = self.tsd.rolling("10S").min()

        nsj1 = pd.Series.to_numpy(tj1)
        nsj2 = pd.Series.to_numpy(tj2)
        nsd = pd.Series.to_numpy(tsd)
        # find indices where course changed by more than 70° in 5S
        indices = np.argwhere(abs(nsj1) > HALF_JIBE_COURSE).flatten()
        # # find indices where course changed by more than 130° in 20S
        # indices_j2 = np.argwhere(abs(nsj2) > FULL_JIBE_COURSE).flatten()
        # # find min speeds > 10 knots:
        # indices_d = np.argwhere(abs(nsd) > MIN_JIBE_SPEED).flatten()
        # # merge: new set with elements common to j and d:
        # indices = list(set(indices_j1) & set(indices_j2) & set(indices_d))
        # record a 20S range around these indices:
        indices_range = [
            set(range(int(i - 10), int(i + 10)))
            for i in indices
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
                    item_iter=k
                )
                results.append(
                    {
                        'description': description,
                        'result': jibe_speed,
                        'n': k,
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
        :param dist: float distance to average speed
        :param n: int number of vmax to record
        :return: TBD list of v[dist]
        """

        def count_time_samples(delta_dist):
            delta_dist = delta_dist[::-1]
            cum_dist = 0
            for i, d in enumerate(delta_dist):
                cum_dist += d
                if cum_dist > dist:
                    break
            return i + 1

        sampling = int(self.sampling.strip("S"))
        min_speed = 7  # m/s min expected speed for max window size
        max_interval = dist / min_speed
        max_samples = int(max_interval / sampling)

        td = self.td.rolling(max_samples).apply(count_time_samples)
        nd = pd.Series.to_numpy(td)
        threshold = min(np.nanmin(nd) + 10, max_samples)
        logger.info(
            f"\nsearching {n} x v{dist} speed\n"
            f"over a window of {max_samples} samples\n"
            f"and found a min of {np.nanmin(nd)*sampling} seconds\n"
            f"to cover {dist}m"
        )
        indices = np.argwhere(nd <= threshold).flatten()
        indices_range = [set(range(int(i - nd[i]), int(i))) for i in indices]
        speed_list = [(self.tsd[range_i].mean(), range_i) for range_i in indices_range]
        speed_list.sort(key=lambda tup: tup[0], reverse=True)

        k = 1
        total_range = set([])
        results = []
        for speed, speed_range in speed_list:
            if not (speed_range & total_range):
                confidence_report = self.append_result_debug(
                    item_range=self.tsd[speed_range].index,
                    item_description=description,
                    item_iter=k
                )
                results.append(
                    {
                        'description': description,
                        'result': speed,
                        'n': k,
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
            result = round(max(ts), 2)
            # remove this speed range to find others:
            tsd[range_begin:range_end] = 0
            # generate debug report
            confidence_report = self.append_result_debug(
                item_range=self.tsd[range_begin:range_end].index,
                item_description=description,
                item_iter=i
            )
            results.append(
                {
                    'description': description,
                    'result': result,
                    'n': i,
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
        self.df_result_debug.loc[item_range, "has_doppler"] = self.thd[item_range]
        self.df_result_debug.loc[item_range, "speed"] = self.tsd[item_range]
        self.df_result_debug.loc[item_range, "raw_speed"] = self.raw_tsd[item_range]
        self.df_result_debug.loc[item_range, "course"] = self.tc[item_range]
        self.df_result_debug.loc[item_range, "dist"] = self.td[item_range]
        self.df_result_debug.loc[item_range, item_description] = item_iter
        # generate report:
        print(self.thd[self.thd > 0.6][item_range])
        if self.thd.dtype == 'float64':
            doppler_ratio = int(100 * len(self.thd[self.thd > 0.6][item_range].dropna()) / len(self.thd[item_range]))
        else:
            doppler_ratio = int(100*len(self.thd[self.thd==True][item_range].dropna()) / len(self.thd[item_range]))
        in_range_raw_tsd = self.raw_tsd[item_range].dropna()
        sampling_ratio = int(100*len(in_range_raw_tsd) / len(self.tsd[item_range]))
        std = round(self.tsd[item_range].std(), 2)
        confidence_report = dict(
            doppler_ratio=doppler_ratio,
            sampling_ratio=sampling_ratio,
            std=std,
        )
        logger.info(
            f"\nconfidence_report on {item_description} in n={item_iter}:\n"
            f"{confidence_report}"
        )
        return confidence_report

    @log_calls(log_args=True, log_result=True)
    def compile_results(self, config_file=None):
        """
        generate the session performance summary
        load config.yaml file with instructions
        about gps analysis functions to call with args
        and record their result in self.result DataFrame
        :return: pd.DataFrame() self_result
        """
        results = []
        config = load_config(config_file)
        # iterate over the config and call the referenced functions:
        for gps_func, iterations in config.items():
            # the same gps_func key cannot be repeated in the yaml description,
            # so we use an iterations list,
            # in order to call several times the same function with different args if needed:
            for iteration in iterations:
                results += getattr(self, gps_func)(
                    description=iteration["description"], **iteration["args"]
                )

        self.result = pd.DataFrame(data=results)
        return self.result

    @log_calls()
    def select_period(self, begin, end):
        self.df = self.df.loc[begin:end]

    @log_calls()
    def plot_speed(self):
        dfs = pd.DataFrame(index=self.tsd.index)
        dfs["raw_speed"] = self.raw_tsd
        dfs["speed"] = self.tsd
        dfs["speed_no_doppler"] = self.ts
        # dfs["delta_doppler"] = self.ts - self.tsd
        dfs.plot()
        plt.show()

    @log_calls()
    def save_to_csv(self):
        """
        save to csv file the simulation results and infos (debug)
        :return: 3 csv files
            - debug.csv with the full DataFrame after filtering
            - result_debug.csv with the runs details of each result
            - result.csv result summary
        """
        self.df.to_csv("debug.csv")
        result_debug = self.df_result_debug[self.df_result_debug.speed.notna()]
        result_debug.to_csv(f"{self.gpx_path}_result_debug.csv")
        self.result.to_csv(f"{self.gpx_path}_result.csv")

parser = ArgumentParser()
parser.add_argument("-f", "--gpx_filename", nargs="?", type=Path, default=".gpx")
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="increases verbosity for each occurence",
)

if __name__ == "__main__":
    args = parser.parse_args()
    basicConfig(level={0: INFO, 1: DEBUG}.get(args.verbose, DEBUG))
    gpx_jla = TraceAnalysis(args.gpx_filename)
    gpx_jla.compile_results()
    #gpx_jla.plot_speed()
    gpx_jla.save_to_csv()
