from flask import Flask, request  # , redirect, render_temple
#from flask_restplus import Resource, Api, reqparse, fields
from flask_restx import Resource, Api, reqparse, fields
from werkzeug.datastructures import FileStorage
import requests

import json
import os
import logging

from gps_analysis import TraceAnalysis, crunch_data
from utils import gpx_results_to_json, load_results

# ******* define Flask api *******
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
            r = requests.get(file_url, )
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



if __name__ == "__main__":
    # ***** start app server *******
    app.run(debug=True, host="0.0.0.0", port=9999)
