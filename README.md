# GPS trace analysis for nautical activities 

analyze speed performance, extracts doppler speed information 

## use

pip3 install -rrequirements.txt<br>
python3  python3 src/core/gps_analysis.py -f Move_2019_10_06_10_55_09_Planche+à+voile_Surf.gpx<br>

leveraging pandas DataFrame:<br>
- format html so that doppler speed data is recognized  
- import to pandas df all gps fields, including doppler speed  
- reindex and resample data  
- plot and save to csv DataaFrame df.csv & TimeSerie ts.csv   

