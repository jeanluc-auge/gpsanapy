from time import time
from utils import load_config
import pandas as pd
from pandas.plotting import parallel_coordinates, andrews_curves
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
import numpy as np


def build_crunch_df(df, result_types):
    if not result_types:
        return
    df2 = pd.DataFrame(
        data={
            result_type: df[df.description == result_type].result
            for result_type in result_types
            if (df.description == result_type).any()
        }
    )
    return df2


def bokeh_plot(all_results, config_plot_file):
    all_results.astype({"result": "float64"}).dtypes

    all_results.loc[all_results.result < 10, "result"] = np.nan
    config_plot = load_config(config_plot_file)
    result_types = config_plot.get("simple_plot", [])
    all_results2 = all_results.set_index(pd.to_datetime(all_results.date))
    df = build_crunch_df(all_results2, result_types)
    df = df.reset_index()
    source = ColumnDataSource(df)
    tool = HoverTool(
        formatters={"@date": "datetime"},
        tooltips=[
            ("name", "$name"),
            ("date", "@date{%Y%m%d}"),
            ("(x, y)", "($x, $y)"),
            ("value", "@date"),
        ],
        mode="vline",
    )
    p = figure(x_axis_type="datetime", x_axis_label="date", y_axis_label="vmoy>12")
    p.add_tools(tool)
    p.circle(
        x="date",
        y="Vmoy>12",
        size=10,
        source=source,
        color="red",
        name="average speed history",
    )

    # p.xaxis.axis_label = "date"
    # p.yaxis.axis_label = "vmoy>12"

    return p
    # show(p)

def plot_speed(gpsana_client):
    p = figure(x_axis_type="datetime", x_axis_label="time")
    dfs = pd.DataFrame(index=gpsana_client.tsd.index)
    dfs["raw_speed"] = gpsana_client.raw_tsd
    dfs[dfs.raw_speed>55] = 55
    dfs["speed"] = gpsana_client.tsd
    dfs["speed_no_doppler"] = gpsana_client.ts
    dfs = dfs.reset_index()

    source = ColumnDataSource(dfs)
    p.line(x='time', y='raw_speed', source=source, color='blue', legend='unfiltered raw speed')
    p.line(x='time', y='speed', source=source, color='red', legend = 'doppler speed')
    p.line(x='time', y='speed_no_doppler', source=source, color='orange', legend='positional speed')
    return p

def plot_vxs_speed(gpsana_client, s):
    xs = f"{int(s)}S"
    p = figure(x_axis_type="datetime", x_axis_label="time")
    dfs = pd.DataFrame(index=gpsana_client.tsd.index)
    dfs["speed"] = gpsana_client.tsd
    dfs[f"speed_xs"] = gpsana_client.tsd.rolling(xs).mean()
    dfs = dfs.reset_index()

    source = ColumnDataSource(dfs)
    p.line(x='time', y='speed', source=source, color='red', legend='doppler speed')
    p.line(x='time', y=f'speed_xs', source=source, color='blue', legend = f'{xs} rolling speed')
    return p
