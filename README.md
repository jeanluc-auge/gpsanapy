# GPS trace analysis for nautical activities 

analyze speed performance, extracts doppler speed information 

## use

pip3 install -rrequirements.txt<br>
python3  python3 src/core/gps_analysis.py -f Move_2019_10_06_10_55_09_Planche+à+voile_Surf.gpx<br>

The results to report are defined in the config.yaml file and are fully parametrizable:<br>
- list of functions to call
- each function can be called several times with different args
- description to put in the final report (result.csv)
 
Leveraging pandas DataFrame:<br>
- format html so that doppler speed data is recognized  
- import to pandas df all gps fields, including doppler speed  
- reindex and resample data
- filter aginst glitches and orientation 360° turns  
- plot and save results to csv files

    debug.csv: full DataFrame of the session<br>
    result_debug.csv: full details for debug<br>
    result.csv: result summary<br>
       

