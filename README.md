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

## command line use

pip3 install -r requirements.txt<br>
python3 src/core/gps_analysis.py -f author_filename.gpx<br>
author_filename.gpx is the gps session file that you want to analyse.

## options

-  **-f** loop over several gpx files:<br>
python3 src/core/gps_analysis.py -f file_1.gpx file_i.gpx ...<br>
make sure that the different files have different author_... prefix 
- **-rd** loop recursively over all gpx files of a given directory and subdir:<br>
python3 src/core/gps_analysis.py -rd directory_name
- **-p** plot speed, distance and course (orientation) results
- **-c** crunch data with matplotlib graphs. Use history results from csv_results/all_results.csv and can be run without a gpx file 

The results of all the gpx files are ranked and aggregated in the same ranking_results.csv file (see output)

## REST API

pip3 install -r requirements.txt<br>
pip3 install -r flask_requirements.txt<br>
python3 src/core/flask_restplus_server.py<br>
api swagger @ http://127.0.0.1:9999/<br>
2 endpoints are currently available:
- ```/fetch_gpx_file/<path:file_url>```<br>
fetch and analyse file @ file_url<br>
you can test it with the swagger, or<br>
```curl -X POST http://127.0.0.1:9999/gpsana/fetch_gpx_file/<file_url>?support=windsurf -H 'accept: application/json'```<br>
in the url text, use '%3A' for ':' and '%2F' for '/'
- ```/upload_gpx_file```<br>
upload file directly on the server<br>
``curl -X POST http://localhost:9999/gpsana/upload_gpx_file?support=windsurf -H "accept: application/json" -F "file=@<file_path>"``

## REST API DOCKER

- in /docker directory, build image:<br>
``sudo docker build -t gpsana_restplus -f Dockerfile_flask_restx .``<br>
- run image:<br>
``sudo docker run -it --rm -p 80:8080 gpsana_restplus``

## configuration

Analysis parameters can be modified in
/config/config.yaml file 

## output

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
- filter against glitches and orientation 360° turns  
- plot and save results to csv files in *csv_results/* directory:

    * *debug.csv*: full DataFrame of the gpx file<br>
        => deep debug only
        => erased after each gpx file processing (not for use with a list of gpx file to process)
    * *filename_result_debug.csv*: DataFrame zoom on the high scores of the session<br>
        => debug only (not for presentation) <br>
    * *filename_result.csv*: individual result summary for the submitted gpx file<br>
        => sessions summary <br>
    * *all_time_results.csv*: swapping history file to record all time sessions history<br>
        => program internal use (not for presentation) <br>
        => updated at each run
    * *ranking_results.csv*: all time sessions or users history with ranking<br>
        => overall results presentation<br>
        => erased at each run <br>
        
 - *execution.log*: full logging of the run with setlevel=INFO 
       

