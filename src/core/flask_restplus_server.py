import re
import flask
from flask import Flask, request  # , redirect, render_temple
#from flask_restplus import Resource, Api, reqparse, fields
from flask_restx import Resource, Api, reqparse, fields
from werkzeug.datastructures import FileStorage
import requests

import json
import os
import logging
import datetime


from gps_analysis import TraceAnalysis, crunch_data, API_VERSION
from utils import gpx_results_to_json, load_results

# ******* define Flask api *******
server = flask.Flask(__name__)
app = Flask(__name__)
api = Api(app)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# ********* gps api *************
gpsana = api.namespace("gpsana", description="nautical gps analysis functions")

parse = reqparse.RequestParser()
parse.add_argument('file', type=FileStorage, location='files')
UPLOAD_DIR = os.path.join(TraceAnalysis.root_dir, 'gpx_file_upload')
DEFAULT_FILENAME = f'anonymous.gpx'

# Create the upload dir if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def parse_file(filename=DEFAULT_FILENAME):
    args = parse.parse_args()
    uploaded_file = args['file']  # This is FileStorage instance
    file_path = os.path.join(UPLOAD_DIR, filename)
    uploaded_file.save(file_path)
    return file_path

def analyse_file(file_path, support, spot):
    gpsana_client = TraceAnalysis(file_path, support=support, spot=spot)
    r = gpsana_client.run()
    if not r:
        return r.log, 400
    response = {}
    response['results'] = r.payload
    response['status'] = r.log
    # gpx_results = gpsana_client.call_gps_func_from_config()
    # response = gpx_results_to_json(gpx_results)
    response['warning'] = gpsana_client.log_warning_list
    response['info'] = gpsana_client.log_info_list
    return (response, 200)

#curl -v -X GET http://localhost:9999/gpsana/version 
@gpsana.route('/version')
class Results(Resource):
    @api.doc(
        description="Returns the version of the API",
    )
    def get(self):
        return {'version': API_VERSION}

#curl -v -X POST http://localhost:9999/gpsana/upload_gpx_file -H "accept: application/json" -F "file=@2019-04-02-1738.gpx"
@gpsana.route('/upload_gpx_file')
class Results(Resource):
    @gpsana.expect(parse)
    @api.doc(
        params={
            'support': {
                'required': False,
                'default': 'windsurf',
                'type': 'string',
            },
            'spot': {
                'required': False,
                'default': None,
                'type': 'string',
            },
        },
        description="get instantaneous results for a gpx file, they will not be saved and they will not count for overall ranking"
    )
    def post(self):
        support = request.args.get('support')
        spot = request.args.get('spot')
        file_path = parse_file()
        response, status = analyse_file(file_path, support, spot)
        # remove file:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return response, status

    # @api.doc(
    #     description="get overall ranking"
    # )
    # def get(self):
    #     all_results = load_results(TraceAnalysis.results_swap_file)

#curl -X POST http://127.0.0.1:9999/gpsana/fetch_gpx_file/https%3A%2F%2Fraw.githubusercontent.com%2Fplotly%2Fdatasets%2Fmaster%2F2011_february_us_airport_traffic.csv?support=windsurf -H 'accept: application/json'
test_url = 'https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv'
@gpsana.route('/fetch_gpx_file/<path:file_url>')
class Upload(Resource):
    @api.doc(
        params={
            'support':{
                'required': False,
                'default': 'windsurf',
                'type': 'string',
            },
            'spot':{
                'required': False,
                'default': None,
                'type': 'string',
            },
        },
        description="fetch gpx file for overall ranking, chose support in (windsurf, windfoil, kitesurf, kitefoil)"
    )
    def post(self, file_url):
        support = request.args.get('support')
        spot = request.args.get('spot')
        filename = file_url.split('/')[-1]
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            r = requests.get(file_url)
            print(file_url)

            with open(file_path, 'wb') as f:
                f.write(r.content)

        except Exception as e:
            return str(e), 400
        response, status = analyse_file(file_path, support, spot)
        # remove file:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return response, status

def Average(l): 
    avg = sum(l) / len(l) 
    return avg

@gpsana.route('/windr/<path:file_url>')
class Upload(Resource):
    @api.doc(
        params={
            'support':{
                'required': False,
                'default': 'windsurf',
                'type': 'string',
            },
            'spot':{
                'required': False,
                'default': None,
                'type': 'string',
            },
        },
        description="Analyze GPX Stats from file_url for specific support (windsurf, windfoil, kitesurf, kitefoil, wingfoil)"
    )
    def post(self, file_url):
        support = request.args.get('support')
        spot = request.args.get('spot')
        filename = file_url.split('/')[-1]
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            r = requests.get(file_url)
            print(file_url)

            with open(file_path, 'wb') as f:
                f.write(r.content)

        except Exception as e:
            return str(e), 400
        response, status = analyse_file(file_path, support, spot)

        trace_date = datetime.datetime.strptime(response["results"]["date"], "%Y-%m-%d");

        windr_json = {
            "id" : None,
            "version" : 2.0,
            "creator" : response["results"]["creator"],
            "type" : support,
            "gps" : None,
            "traceDate" : {
                "$date": {
                    "$numberLong": response["results"]["datetime"]*1000
                }
            },
            "date" :{
                "day" : trace_date.day,
                "month" : trace_date.month,
                "year" : trace_date.year,
            },
            "location":{
                "latitude" : response["results"]["location_lat"],
                "longitude" : response["results"]["location_lon"]
            },      
            "windDirection" : "",
            "totalLength" : 0,
            "averageSpeed" : 0,
            "duration" : response["results"]["duration"],
            "source" : {
                "type" : "speedResults",
                "version" : API_VERSION,
                "creator" : "GPSAnaPy",
                "integration" : None,
                "computation" : None,
                "software" : {
                    "name" : "GPSAnaPy",
                    "version" : API_VERSION,
                    "url" : "https//github.com/windr-app/gpsanapy",
                },
            },
            "board" : {
                "quiver_id": None,
                "catalog_id": None,
                "brand": None,
                "model": None,
                "volume": None,
                "year": None,
            },
            "sail" : {                    
                "quiver_id" : None,
                "catalog_id" : None, 
                "brand" : None, 
                "model" : None,
                "size" : None,
                "year" : None
            },               
            "fin":{ 
                "quiver_id" : None,
                "catalog_id" : None,
                "brand" : None,
                "model" : None,
                "size" : None,
                "year" : None,
            },      
        };

        perf_analysis = response["results"]["perf_analysis"];

        runs_1_s = [];
        runs_2_s = [];
        runs_10_s = [];
        runs_20_s = [];
        runs_1800_s = [];
        runs_3600_s = [];
        runs_100_m = [];
        runs_250_m = [];
        runs_500_m = [];
        runs_1000_m = [];
        runs_1852_m = [];
        alpha_250_runs = [0.0,0.0,0.0,0.0,0.0];
        alpha_500_runs = [0.0,0.0,0.0,0.0,0.0];
        
        totalLength = 0.0;
        averageSpeed = 0.0;

        jibes_vmax = [];

        ## loop on perf_analysis
        for item in perf_analysis:
            if ( item["description"] == "vmax_1s" ) :
                runs_1_s.append(item["result"]);
            elif ( item["description"] == "vmax_2s" ) :
                runs_2_s.append(item["result"]);
            elif ( item["description"] == "vmax_10s" ) :
                runs_10_s.append(item["result"]);
            elif ( item["description"] == "vmax_20s" ) :
                runs_20_s.append(item["result"]);
            elif ( item["description"] == "v_30mn" ) :
                runs_1800_s.append(item["result"]);
            elif ( item["description"] == "v_1h" ) :
                runs_3600_s.append(item["result"]);                
            elif ( item["description"] == "vmax_100m" ) :
                runs_100_m.append(item["result"]);
            elif ( item["description"] == "vmax_250m" ) :
                runs_250_m.append(item["result"]);                
            elif ( item["description"] == "vmax_500m" ) :
                runs_500_m.append(item["result"]);
            elif ( item["description"] == "vmax_1000m" ) :
                runs_1000_m.append(item["result"]);
            elif ( item["description"] == "vmax_1852m" ) :
                runs_1852_m.append(item["result"]);                
            elif ( item["description"] == "vmax_jibe" ) :
                jibes_vmax.append(item["result"]);                
            elif ( item["description"] == "planning_distance>0" ) :
                totalLength = item["result"];   
            elif ( item["description"] == "Vmoy>0" ) :
                averageSpeed = item["result"];   
            else : 
                print(item)

        windr_json["totalLength"] = totalLength;
        windr_json["averageSpeed"] = averageSpeed;
        

        windr_json["run_1_s_avg"] = Average(runs_1_s);    
        windr_json["runs_1_s"] = runs_1_s;    
        windr_json["run_2_s_avg"] = Average(runs_2_s);    
        windr_json["runs_2_s"] = runs_2_s; 
        windr_json["run_10_s_avg"] = Average(runs_10_s);    
        windr_json["runs_10_s"] = runs_10_s; 
        windr_json["run_20_s_avg"] = Average(runs_20_s);    
        windr_json["runs_20_s"] = runs_20_s;         
        windr_json["run_1800_s_avg"] = Average(runs_1800_s);    
        windr_json["runs_1800_s"] = runs_1800_s;    
        windr_json["run_3600_s_avg"] = Average(runs_3600_s);    
        windr_json["runs_3600_s"] = runs_3600_s;           
        windr_json["run_100_m_avg"] = Average(runs_100_m);    
        windr_json["runs_100_m"] = runs_100_m; 
        windr_json["run_250_m_avg"] = Average(runs_250_m);    
        windr_json["runs_250_m"] = runs_250_m;         
        windr_json["run_500_m_avg"] = Average(runs_500_m);    
        windr_json["runs_500_m"] = runs_500_m; 
        windr_json["run_1000_m_avg"] = Average(runs_1000_m);    
        windr_json["runs_1000_m"] = runs_1000_m; 
        windr_json["run_1852_m_avg"] = Average(runs_1852_m);    
        windr_json["runs_1852_m"] = runs_1852_m; 

        # TODO : Implement alpha 250 and 500
        windr_json["alpha_250_avg"] = Average(alpha_250_runs);    
        windr_json["alpha_250_runs"] = alpha_250_runs; 
        windr_json["alpha_500_avg"] = Average(alpha_500_runs);    
        windr_json["alpha_500_runs"] = alpha_500_runs; 

        windr_json["jibes_vmax_avg"] = Average(jibes_vmax);
        windr_json["jibes_vmax"] = jibes_vmax;


        # remove file:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return windr_json, status

if __name__ == "__main__":
    # ***** start app server *******
    app.run(debug=True, host="0.0.0.0", port=9999)