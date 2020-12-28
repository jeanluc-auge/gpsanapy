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


from utils import log_calls, reindex, resample, TraceAnalysisException

logger = getLogger()


class TraceAnalysis:
    def __init__(self, gpx_path, sampling="1S"):
        logger.info(f"init {self.__class__.__name__} with file {gpx_path}")
        self.sampling = sampling
        self.gpx_path = gpx_path
        self.df = self.load_df(gpx_path)
        begin = "2019-04-02 16:34:00+00:00"
        end = "2019-04-02 16:36:00+00:00"
        # self.select_period(begin, end)
        self.clean_df()

        # generate key time series with fixed sampling and interpolation:
        self.tsd = resample(self.df, sampling, "speed")
        self.ts = resample(self.df, sampling, "speed_no_doppler")
        self.td = resample(self.df, sampling, "delta_dist")
        self.save_to_csv()

    def load_df(self, gpx_path):
        html_soup = self.load_gpx_file_to_html(gpx_path)
        tracks = self.format_html_to_gpx(html_soup)
        df = self.to_pandas(tracks[0].segments[0])
        # TODO filter for delta speed and delta_dist > 30
        df = reindex(df, "time")
        return df

    @log_calls()
    def load_gpx_file_to_html(self, gpx_path):
        with open(gpx_path, "r") as gpx_file:
            html_soup = BeautifulSoup(gpx_file, "html.parser")
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
        split_data = [
            {
                "lon": point.longitude,
                "lat": point.latitude,
                "time": point.time,
                "speed": (point.speed if point.speed else raw_data.get_speed(i)),
                "speed_no_doppler": raw_data.get_speed(i),
                "course": point.course,
                "has_doppler": bool(point.speed),
                "delta_dist": (
                    point.distance_2d(raw_data.points[i - 1]) if i >= 1 else 0
                ),
            }
            for i, point in enumerate(raw_data.points)
        ]
        df = pd.DataFrame(split_data)
        if not df.has_doppler.all():
            ts = pd.Series(data=0, index=df.index)
            ts[df.has_doppler == True] = 1
            doppler_ratio = int(100 * sum(ts) / len(ts))
            if doppler_ratio < 50:
                raise TraceAnalysisException(
                    f"doppler speed is available on only {doppler_ratio}% of data"
                )
            logger.warning(
                f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"Doppler speed is not available on all sampling points\n"
                f"Only {doppler_ratio}% of the points have doppler data\n"
                f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )
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
            if x[0] > 0.3:
                exiting = False
                for i, a in enumerate(x):
                    # find spike interval length by searching negative spike end
                    if a < -0.1:
                        exiting = True
                    elif exiting:
                        break
            return i

        # add a new column with total elapsed time in seconds:
        self.df["elapsed_time"] = pd.to_timedelta(
            self.df.index - self.df.index[0]
        ).astype("timedelta64[s]")
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
        self.df.dropna(inplace=True)
        self.df.to_csv("debug.csv")

        return len(indices) > 0

    def clean_df(self):
        """
        filter self.df on speed_no_doppler and speed fields
        to remove acceleration spikes > 0.3g
        :return: modify self.df
        """
        # calculate and create column of elapsed time in seconds:
        # convert ms-1 to knots:
        self.df.speed = round(self.df.speed * 1.94384, 2)
        self.df.speed_no_doppler = round(self.df.speed_no_doppler * 1.94384, 2)

        erratic_data = True
        iter = 1
        while erratic_data and iter < 10:
            erratic_data = self.filter_on_field("speed_no_doppler")
            erratic_data = erratic_data or self.filter_on_field("speed")
            iter += 1
        # tf = resample(self.df, "1S", "speed_acceleration")
        # tf.plot()
        # ta = resample(self.df, "1S", "speed_filtering")
        # ta.plot()
        self.df.to_csv("debug.csv")

    @log_calls()
    def speed_dist(self, dist=500, n=5):
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
            return i

        sampling = int(self.sampling.strip('S'))
        min_speed = 7 # m/s min expected speed for max window size
        max_interval = dist / min_speed
        max_samples = int(max_interval / sampling)

        ts = self.tsd.rolling(max_samples).apply(count_time_samples)
        ns = pd.Series.to_numpy(ts)
        threshold = min(np.nanmin(ns)+10, max_samples)
        logger.info(
            f"\nsearching {n} x v{dist} speed\n"
            f"over a window of {max_samples} samples\n"
            f"and found a {np.nanmin(ns)*sampling} seconds min length\n"
            f"resulting in a treshold length of {threshold} samples"
        )
        k = 0
        total_range = set([])
        result = []
        indices = np.argwhere(ns <= threshold).flatten()
        indices_range = [set(range(int(i-ns[i]), int(i))) for i in indices]
        speed_list = [
            (
                self.tsd[range_i].mean(),
                range_i
             )
            for range_i in indices_range
        ]
        speed_list.sort(key = lambda tup: tup[0], reverse=True)
        for x in speed_list:
            if not (x[1] & total_range):
                result.append(x[0])
                total_range = total_range | x[1]
                k+=1
            if k >= n:
                break

        print(result)
        #ts.index.get_loc(ts2.idxmin('index'))
        ts.plot()
        plt.show()
        return result

    @log_calls(log_args=True, log_result=False)
    def speed_xs(self, s=10, n=10):
        """
        Vmax: n * x seconds
        :param xs: int time interval in seconds
        :param n: number of records
        :return: TBD
        """
        xs = str(s) + "S"

        # calculate s seconds Vmax on all data:
        ts = self.tsd.rolling(xs).mean()

        # select n best Vmax:
        nxs_list = []
        for i in range(1, n + 1):
            range_end = ts.idxmax()
            # range_end = ts.index[ts==max(ts)][0]
            range_begin = range_end - datetime.timedelta(seconds=s)
            ts[range_begin:range_end] = 0
            nxs_list.append((range_begin, range_end, round(max(ts), 2)))

        nxs_speed_results = "\n".join([f"{x}-{y}: {z}" for x, y, z in nxs_list])
        logger.info(
            f"\n===============================\n"
            f"Best vmax {n} x {s} seconds\n"
            f"{nxs_speed_results}"
            f"\n===============================\n"
        )
        return nxs_speed_results

    @log_calls()
    def select_period(self, begin, end):
        self.df = self.df.loc[begin:end]

    @log_calls()
    def plot_speed(self):
        dfs = pd.DataFrame(index=self.ts.index)
        dfs["speed"] = self.tsd
        dfs["speed_no_doppler"] = self.ts
        # dfs["delta_doppler"] = self.ts - self.tsd
        dfs.plot()
        plt.show()

    @log_calls()
    def save_to_csv(self, df_file="df.csv", ts_file="ts.csv"):
        self.df.to_csv(df_file)
        self.ts.to_csv(ts_file)


# code Nunu: ==================================================


class GpxFileAnalyse:
    def __init__(self, gpx_path):
        self.file_path = gpx_path
        gpx_file = open(gpx_path, "r")
        logger.info(f"init gpx analysis with file {gpx_path}")
        self.gpx = gpxpy.parse(gpx_file)
        self.points_tab = []
        self.total_number_of_point = 0
        self.last_index = 0
        self.parse_result = False
        self.total_number_of_point = 0
        self.fill_tab()
        self.time_spread = 1

    def fill_tab(self):

        #
        #   Fill tab with point information
        #
        distance = 0
        heure_offset = 0
        heure_prev = 0
        distance_previous = 0
        previous_track_point = None
        time_point_previous = 0
        for track in self.gpx.tracks:
            for segment in track.segments:
                for point_no, point in enumerate(segment.points):
                    if point_no == 0:
                        previous_track_point = point
                    if point_no > 0:
                        speed = int(0)
                        if point.speed != None:
                            speed = point.speed

                        else:
                            speed = point.speed_between(segment.points[point_no - 1])

                        if speed != None:
                            self.parse_result = True
                            #
                            # get speed of the point
                            #
                            speed_point = round(speed * 1.94384, 2)
                            point.speed_between(point)
                            #
                            # get time of the point
                            #
                            heure = point.time.strftime("%H")
                            minute = point.time.strftime("%M")
                            second = point.time.strftime("%S")

                            if heure_offset == 0 and int(heure) < int(heure_prev):
                                # passage 24h -> 0
                                heure_offset = 24
                            heure_prev = heure
                            heure = int(heure_offset) + int(heure)

                            time_point = (
                                int(heure) * 3600 + int(minute) * 60 + int(second)
                            )
                            #
                            # Get distance from the beginning
                            #
                            distance_previous_point = round(
                                point.distance_2d(previous_track_point), 2
                            )
                            distance = round(distance + distance_previous_point, 2)
                            previous_track_point = point

                            #
                            #   Check the point validity & fill the tab
                            #

                            # a point with more that 2s with the previous one is not valable
                            # test the distance made in less than 2s
                            if not (
                                time_point - time_point_previous > 10
                                or point.distance_2d(previous_track_point) > 30
                            ):
                                # print "-- %s %s" % ( (time_point - time_point_previous  ),distance_previous)
                                # else:
                                self.points_tab.append(
                                    (
                                        time_point,
                                        speed_point,
                                        point.time,
                                        distance,
                                        distance_previous_point,
                                    )
                                )
                                distance_previous = distance
                            # print("speed point %s" % speed_point)
                            time_point_previous = time_point

                break
            break

        self.total_number_of_point = len(self.points_tab)
        self.last_index = self.total_number_of_point - 1

        if self.parse_result == False:
            print("Error parsing file : %s" % self.file_path)


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

    # logger.debug(
    #     f"\n==========================\n"
    #     f"now testing code of nunu:"
    #     f"\n==========================\n"
    # )
    # gpx_nunu = GpxFileAnalyse(args.gpx_filename)
    # # display one item / line:
    # dis = '\n'.join([str(x) for x in gpx_nunu.points_tab])
    # logger.debug(f"nunu code result: points_tab {dis}")

    logger.info(
        f"\n==========================\n"
        f"now testing code of jla:"
        f"\n==========================\n"
    )
    gpx_jla = TraceAnalysis(args.gpx_filename)
    logger.info(f"jla code result: {gpx_jla.df}")
    #gpx_jla.plot_speed()
    gpx_jla.speed_xs()
    gpx_jla.speed_dist()
