# GPS trace analysis for nautical activities 

analyze speed performance, extracts doppler speed information out of .gpx session files.<br>
Features:<br>
- n * vmax over x seconds
- n * vmax over x meter
- vmin of n * best jibes
- vmoy > vmin (average cruising speed)
- total distance > vmin (planning distance)
- % distance > vmin on distance (true planning ratio)
- % distance > vmin on time (planning ratio on time)

## use

pip3 install -rrequirements.txt<br>
python3 src/core/gps_analysis.py -f author_filename.gpx<br>
author_filename.gpx is the gps session file that you want to analyse.

The results to report are defined in the config.yaml file and are fully parametrizable:<br>
- list of functions to call
- each function can be called several times with different args
- description to put in the final report (result.csv)
- ranking_group to calculate overall ranking based on groups
- if the yaml config file is modified, the all time ranking csv reports are reset (a all_time_results.old copy is saved)
 
Leveraging pandas DataFrame:<br>
- format html so that doppler speed data is recognized  
- import to pandas df all gps fields, including doppler speed  
- reindex and resample data
- filter aginst glitches and orientation 360° turns  
- plot and save results to csv files

    * debug.csv: full DataFrame of the session<br>
    * result_debug.csv: DataFrame zoom on the high scores of the session<br>
        => debug only (not for presentation) <br>
        => erased at each run
    * result.csv: individual result summary for the submitted gpx file<br>
        => sessions summary <br>
        => erased at each run
    * all_time_results.csv: swapping history file to record all time sessions history<br>
        => program internal use (not for presentation) <br>
        => updated at each run
    * ranking_results.csv: all time sessions or users history with ranking<br>
        => overall results prensentation
        => erased at each run <br>
       

