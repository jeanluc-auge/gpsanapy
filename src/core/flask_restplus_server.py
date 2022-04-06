from flask import Flask, request  # , redirect, render_temple
#from flask_restplus import Resource, Api, reqparse, fields
from flask_restx import Resource, Api, reqparse, fields
from werkzeug.datastructures import FileStorage

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

#curl -v -X GET "http://localhost:9999/gpsana/gpsana/crunch" -H "accept: application/json"
@gpsana.route('/crunch')
class DataCrunch(Resource):
    @api.doc(description=crunch_data.__doc__)
    def get(self):
        crunch_data()
        return (
            "data crunched", 200
        )

#curl -v -X POST http://localhost:9999/gpsana/upload -H "accept: application/json" -F "file=@2019-04-02-1738.gpx"
@gpsana.route('/results')
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
        args=parse.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        file_path = os.path.join(UPLOAD_DIR, f'anonymous.gpx')
        uploaded_file.save(file_path)
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
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        return response, 200

    # @api.doc(
    #     description="get overall ranking"
    # )
    # def get(self):
    #     all_results = load_results(TraceAnalysis.results_swap_file)


@gpsana.route('/upload_gpx_file/<string:user>')
class Upload(Resource):
    @gpsana.expect(parse)
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
        description="upload gpx file for overall ranking, chose support in (windsurf, windfoil, kitesurf, kitefoil)"
    )
    def post(self, user):
        support = request.args.get('support')
        spot = request.args.get('spot')
        args=parse.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        file_path = os.path.join(UPLOAD_DIR, f'{user}.gpx')
        uploaded_file.save(file_path)
        gpsana_client = TraceAnalysis(file_path, support=support, spot=spot)
        gpx_results = gpsana_client.call_gps_func_from_config()
        gpsana_client.rank_all_results(gpx_results)
        gpsana_client.save_to_csv(gpx_results)
        return gpx_results_to_json(gpx_results), 200



if __name__ == "__main__":
    # ***** start app server *******
    app.run(debug=True, host="0.0.0.0", port=9999)
