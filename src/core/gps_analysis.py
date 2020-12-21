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
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


from utils import log_calls, reindex, resample

logger = getLogger()

class TraceAnalysis():

    def __init__(self, gpx_path):
        logger.info(f"init {self.__class__.__name__} with file {gpx_path}")
        self.gpx_path = gpx_path
        self.html_soup = self.load_gpx_file_to_html()
        self.tracks = self.format_html_to_gpx()
        self.df = self.to_pandas(self.tracks[0].segments[0])
        begin = "2019-10-06 09:56:00+00:00"
        end = "2019-10-06 09:57:00+00:00"
        #self.select_period(begin, end)
        self.df = reindex(self.df, 'time')
        self.ts = resample(self.df, "10S")
        self.save_to_csv()

    @log_calls
    def clean_gpx(self, filename):
        with open(filename, 'r') as gpx_file:
            soup = BeautifulSoup(gpx_file, 'html.parser')

    @log_calls()
    def load_gpx_file_to_html(self):
        with open(self.gpx_path, 'r') as gpx_file:
            html_soup = BeautifulSoup(gpx_file, 'html.parser')
        return html_soup

    @log_calls()
    def format_html_to_gpx(self):
        """
        remove unwanted tags in html file:
        - <extensions>
        - <gpxdata:speed>
        and replace them with <speed> tag

        :return: gps tracks
        """

        # add <speed> tag:
        for el in self.html_soup.findAll('gpxdata:speed'):
            el.wrap(self.html_soup.new_tag('speed'))
        # remove <gpxdata:speed> tag:
        for el in self.html_soup.findAll('gpxdata:speed'):
            el.unwrap()
        # remove <extensions> tag:
        for el in self.html_soup.findAll('extensions'):
            el.unwrap()

        gpx = gpxpy.parse(str(self.html_soup))
        tracks = gpx.tracks
        return tracks

    @log_calls(log_args=False, log_result=True)
    def to_pandas(self, raw_data):
        split_data = [
            {
                'lon':point.longitude,
                'lat': point.latitude,
                'time': point.time,
                'speed': (
                    point.speed if point.speed else round(raw_data.get_speed(i)* 1.94384, 2)
                ),
                'speed_no_doppler': round(raw_data.get_speed(i)* 1.94384, 2),
                'has_doppler': bool(point.speed),
                'delta_dist': (point.distance_2d(raw_data.points[i-1]) if i>=1 else 0)
            }
            for i,point in enumerate(raw_data.points)
        ]
        df = pd.DataFrame(split_data)
        # TODO filter for delta speed and delta_dist > 30
        return df

    @log_calls()
    def select_period(self, begin, end):
        self.df = self.df.loc[begin:end]

    @log_calls()
    def plot_speed(self):
        self.ts.plot()
        plt.show()

    @log_calls()
    def save_to_csv(self, df_file="df.csv", ts_file="ts.csv"):
        self.df.to_csv(df_file)
        self.ts.to_csv(ts_file)

# code Nunu: ==================================================

class GpxFileAnalyse:
    def __init__(self, gpx_path):
        self.file_path = gpx_path
        gpx_file = open(gpx_path, 'r')
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
        heure_offset=0
        heure_prev=0
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
                            heure_prev=heure
                            heure = int(heure_offset) + int(heure)

                            time_point = int(heure) * 3600 + int(minute) * 60 + int(second)
                            #
                            # Get distance from the beginning
                            #
                            distance_previous_point = round(point.distance_2d(previous_track_point), 2)
                            distance = round(distance + distance_previous_point, 2)
                            previous_track_point = point

                            #
                            #   Check the point validity & fill the tab
                            #

                            # a point with more that 2s with the previous one is not valable
                            # test the distance made in less than 2s
                            if not (time_point - time_point_previous > 10 or point.distance_2d(
                                    previous_track_point) > 30):
                                # print "-- %s %s" % ( (time_point - time_point_previous  ),distance_previous)
                               # else:
                                self.points_tab.append(
                                    (time_point, speed_point, point.time, distance, distance_previous_point))
                                distance_previous = distance
                            #print("speed point %s" % speed_point)
                            time_point_previous = time_point

                break
            break

        self.total_number_of_point = len(self.points_tab)
        self.last_index = self.total_number_of_point - 1

        if self.parse_result == False:
            print("Error parsing file : %s" % self.file_path)

parser = ArgumentParser()
parser.add_argument(
    "-f", "--gpx_filename", nargs="?", type=Path, default='.gpx'
)
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

    logger.debug(
        f"\n==========================\n"
        f"now testing code of nunu:"
        f"\n==========================\n"
    )
    gpx_nunu = GpxFileAnalyse(args.gpx_filename)
    # display one item / line:
    dis = '\n'.join([str(x) for x in gpx_nunu.points_tab])
    logger.debug(f"nunu code result: points_tab {dis}")

    logger.info(
        f"\n==========================\n"
        f"now testing code of jla:"
        f"\n==========================\n"
    )
    gpx_jla = TraceAnalysis(args.gpx_filename)
    logger.info(f"jla code result: {gpx_jla.df}")
    gpx_jla.plot_speed()