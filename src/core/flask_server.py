from flask import Flask, request, g, render_template, redirect, flash, url_for, session, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import joinedload
#from flask import reqparse, fields, Resource, Api,
from werkzeug.datastructures import FileStorage
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import json
import sqlite3
import os
import logging
import functools
from copy import deepcopy
from pathlib import Path
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from gps_analysis import TraceAnalysis, Trace, crunch_data
from utils import gpx_results_to_json, load_results, split_path
# from db import init_app, init_db
from visu import bokeh_plot, bokeh_speed, bokeh_speed_density, all_results_speed_density, compare_all_results_density
from bokeh.models.callbacks import CustomJS
from bokeh.embed import components
from bokeh.resources import INLINE

UPLOAD_FOLDER = '/home/jla/gps/gpx_file_upload'
ALLOWED_EXTENSIONS = {'.gpx', '.sml'}
SUPPORT_CHOICE = ['all', 'windsurf', 'windfoil', 'kitesurf', 'kitefoil', 'kayak']
SPOT_CHOICE = ['all', 'g13', 'nantouar', 'trestel', 'keriec', 'bnig', 'other']
# check or create that file upload and database dir exist:
dir_path = [os.path.join(TraceAnalysis.root_dir, d) for d in ('database', 'gpx_file_upload')]
for d in dir_path:
    if not Path(d).is_dir():
        os.makedirs(d)
# ******* define Flask api *******
app = Flask(__name__, instance_relative_config=True, static_url_path='/static', static_folder='static')

# *** sql ****
# engine = create_engine('sqlite:///:memory:', echo=True)
# Base = declarative_base()
app.config.from_mapping(SECRET_KEY='dev')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/jla/gps/database/test.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
trace =  Trace()
#Locquirec 3.6451W 48.6541N
#Pleubian 3.1396W 48.8424N
#Lancieux 2.1491W 48.6094N
#Loudeac 2.7520W 48.1763N
#22 3.645W-2.15W 49N-48.1N
REGIONS = {
    'wb22': {'lon': (2.15, 3.645), 'lat': (48.1, 49)},
}

class User(db.Model): #(Base)
     #__tablename__ = 'users'
     id = db.Column(db.Integer, primary_key=True)
     username = db.Column(db.String(50), unique=True, nullable=False)
     email = db.Column(db.String(128), unique=True, nullable=False)
     location = db.Column(db.String(128))
     infos = db.Column(db.String(128))
     password = db.Column(db.Text, nullable=False)
     def __repr__(self):
        return f"<User(username={self.username}, password={self.password})>"

class TraceFiles(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    approved = db.Column(db.Boolean)
    filename = db.Column(db.String(255), unique=True, nullable=False) # stem: no extension, no path
    file_path = db.Column(db.String(255), unique=True, nullable=False) # absolute path
    spot = db.Column(db.String(255), nullable=False)
    support = db.Column(db.String(50), nullable=False)
    user = db.relationship('User', backref=db.backref('gpxfiles', lazy='joined'))

#Base.metadata.create_all(engine)
#User.query.delete()
#TraceFiles.query.delete()
with app.app_context():
    db.create_all()
    # cdncred = CdnCredentials(id=1, auth_token='test', cdn_username='cdngw_cdngw@orange.com', cdn_password='system12')
    # cdncred.save_to_db()
    # cust = CustomerModel(id=1, username='cdngw', scope='write', email='cdngw@orange.com', organization='application',
    #                      cdncredentialid=1)
    # cust.save_to_db()

# df = pd.DataFrame({'name' : ['User 1', 'User 2', 'User 3']})
# df.to_sql(name='cdc', con=db.engine, if_exists='append')

# ed_user = User(username='ed', password=generate_password_hash('r'))
# print(1111111111111111111,ed_user, ed_user.id)

#Session = sessionmaker(bind=engine)
#session = Session()
# db.session.add(ed_user)
# db.session.add_all([
#     User(name='wendy', fullname='Wendy Williams', nickname='windy'),
#     User(name='mary', fullname='Mary Contrary', nickname='mary'),
#     User(name='fred', fullname='Fred Flintstone', nickname='freddy')
# ])
# db.session.commit()

# our_user = db.session.query(User).filter_by(name='ed').first()
#
# print(111111111111,our_user, ed_user.id)
# # test rollback
# ed_user.name = 'Edwardo'
# fake_user = User(name='fakeuser', fullname='Invalid', nickname='12345')
# db.session.add(fake_user)
# db.session.rollback()
#
# for name in db.session.query(User.name).filter_by(name='ed'):
#     print(222222222,name)
# our_users = db.session.query(User).filter(User.name.in_(['ed', 'fakeuser'])).all()
# _ = db.session.query(User).filter(User.username=='ed').first()
# _ = [u.username for u in db.session.query(User).all()]

#
# print(2222222222,our_users)
# # query:
# for instance in db.session.query(User).order_by(User.id)[1:2]:
#     print('000000000000',instance.name, instance.fullname)
# for instance in db.session.query(User.name, User.nickname):
#     print(11111111111111,instance.name, instance.nickname)

# @app.teardown_appcontext
# def close_connection(exception):
#     db = getattr(g, '_database', None)
#     if db is not None:
#         db.close()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash("you need to login to perform this action\n or register if you don't have an account ")
            return redirect(url_for('login'))

        return view(**kwargs)

    return wrapped_view

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = db.session.query(User).filter(User.id==user_id).first()

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        location = request.form['location']
        infos = request.form['infos']
        #db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif not email:
            error = 'Email is required.'
        elif db.session.query(User).filter(User.username==username).first() is not None:
            error = 'User {} is already registered.'.format(username)

        if error is None:
            new_user = User(
                username=username,
                password=generate_password_hash(password),
                email=email,
                location=location,
                infos=infos
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))

        flash(error)

    return render_template('/register.html')

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        #db = get_db()
        error = None
        user = db.session.query(User).filter(User.username==username).first()

        if user is None:
            error = 'Incorrect username.'

        elif not check_password_hash(user.password, password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('index'))

        flash(error)

    return render_template('/login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def get_file_attr(filename):
    """
    return file status, file stem, file path and check it is valid
    :param filename: uploaded filename
    :return: (status (bool), filename wo extension, full file path)
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    filename, file_extension = split_path(filename)
    logger.info(f"uploading file {filename} with extension {file_extension} and path {file_path}")
    if db.session.query(TraceFiles).filter(TraceFiles.filename == filename).first() is not None:
        flash(f"file {filename} already exists")
        return False, None, None
    if file_extension not in ALLOWED_EXTENSIONS:
        flash(f"file extension {file_extension} is not supported")
        return False, None, None
    return True, filename, file_path
    # return '.' in filename and \
    #        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file(id, check_author=True):
    file = db.session.query(TraceFiles).filter(TraceFiles.id == id).first()
    if file is None:
        abort(404, f"file id {id} doesn't exist.")
    if check_author and file.user_id != g.user.id and g.user.username!='admin':
        abort(403)

    return file

def get_form_required(item_name):
    item = request.form[item_name]
    if not item:
        flash(f'No {item_name} part')
        return redirect(request.url)
    return item

def gen_bokeh_resources(p):
    bokeh_resources = {}
    bokeh_resources['js_resources'] = INLINE.render_js()
    bokeh_resources['css_resources'] = INLINE.render_css()
    bokeh_resources['script'], bokeh_resources['div'] = components(p)
    return bokeh_resources

# ********* gps flask server *************

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/files')
def files():
    gpxfiles = db.session.query(TraceFiles).order_by(TraceFiles.id).all()
    return render_template('files.html', gpxfiles=gpxfiles)

@app.route('/upload_file', methods=('GET', 'POST'))
@login_required
def upload_file():
    if request.method == 'POST':
        spot = get_form_required('spot')
        spot_infos = request.form['spot_infos']
        support = get_form_required('support')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if not file:
            flash('file not valid')
            return redirect(request.url)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        status, filename, file_path = get_file_attr(filename)
        if not status:
            return redirect(request.url)

        file.save(file_path)
        new_file = TraceFiles(
            filename=filename,
            file_path=file_path,
            spot=spot,
            support=support,
            user_id=g.user.id,
            approved=False,
        )
        db.session.add(new_file)
        db.session.commit()
        # pre-analyse file (convert gpx to df and save to parquet for future analysis):
        gpsanaclient = TraceAnalysis(
            file_path,
            support=new_file.support,
            spot=new_file.spot,
            author=new_file.user.username,
            parquet_loading=False
        )
        status, error = gpsanaclient.run()
        if not status:
            flash('\n'.join(error))
        else:
            db.session.add(new_file)
            db.session.commit()
        return redirect(url_for('files'))

    return render_template('upload_file.html', support_choice=SUPPORT_CHOICE, spot_choice=SPOT_CHOICE)

@app.route('/<int:id>/delete_file', methods=('GET', 'POST'))
@login_required
def delete_file(id):
    file = get_file(id) # check it exists
    db.session.delete(file)
    db.session.commit()
    trace.delete_result(file.filename, file.file_path)
    return redirect(url_for('files'))

@app.route('/reload_files', methods=('GET', 'POST'))
@login_required
def reload_files():
    files = db.session.query(TraceFiles).order_by(TraceFiles.id).all()
    error_dict = {}
    for file in files:
        gpsanaclient = TraceAnalysis(file.file_path, author=file.user.username, spot=file.spot, support=file.support)
        status, error = gpsanaclient.run()
        if not status:
            error_dict[gpsanaclient.filename] = error
    if error_dict:
        flash(error_dict)
    return redirect(url_for('files'))

@app.route('/<int:id>/analyse')
def analyse(id):
    # analyse file
    file = get_file(id, check_author=False)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    gpsana_client = TraceAnalysis(file_path, author=file.user.username, spot=file.spot, support=file.support)
    status, error = gpsana_client.run()
    if not status:
        return render_template(
            'analyse.html',
            response={},
            warnings=error,
            infos={},
            file=file,
            bokeh_template={},
        ).encode(encoding='UTF-8')

    reduced_results = trace.reduced_results(
            #by_support=file.support,
            by_author=file.user.username,
        )
    response = gpx_results_to_json(gpsana_client.gpx_results)
    warnings = gpsana_client.log_warning_list
    infos = gpsana_client.log_info_list
    # bokeh
    bokeh_template = {}
    # plot speeds
    p = bokeh_speed(gpsana_client)
    # grab the static resources
    bokeh_template['plot speed data'] = gen_bokeh_resources(p)
    # plot rolling speed
    s = 10
    p = bokeh_speed_density(gpsana_client, s)
    # grab the static resources
    bokeh_template[f'session speed distribution'] = gen_bokeh_resources(p)

    p = compare_all_results_density(reduced_results, gpsana_client, ['vmax_10s', 'vmax_jibe'])
    bokeh_template[f'compare session results (vertical lines) with {file.user.username} all time results distributions'] = gen_bokeh_resources(p)

# render template

    return render_template(
        'analyse.html',
        response=response,
        warnings=warnings,
        infos=infos,
        file=file,
        bokeh_template=bokeh_template,
    ).encode(encoding='UTF-8')

@app.route('/ranking', methods=('GET', 'POST'))
@app.route('/ranking/<string:support>', methods=('GET', 'POST'))
def ranking(support=SUPPORT_CHOICE[0]):
    df = trace.rank_all_results(by_support=support)
    if df is None:
        flash(f"there is not enough data in {support} to rank")
        return redirect(url_for('ranking', support=SUPPORT_CHOICE[0]))

    if request.method == 'POST':
        support = get_form_required('support')
        print(support)
        return redirect(url_for('ranking', support=support))

    return render_template(
        '/ranking.html',
        support_choice=SUPPORT_CHOICE,
        tables = [df.to_html(classes='data')],
        titles = df.columns.values,
        support = support
    )

@app.route('/crunch_data', methods=('GET', 'POST'))
@app.route('/crunch_data/<string:by_support>/<string:by_spot>/<string:by_author>',  methods=('GET', 'POST'))
def crunch_data(by_support='all', by_spot='all', by_author='all'):
    reduced_results = trace.reduced_results(
            by_support=by_support,
            by_spot=by_spot,
            by_author=by_author,
            check_config=True
        )
    print(reduced_results.head())
    if reduced_results is None or reduced_results.empty:
        flash(f"there is not enough data in {by_support} support, {by_spot} spot and {by_author} author to rank")
    p = all_results_speed_density(reduced_results)
    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    script, div = components(p)

    if request.method == 'POST':
        by_spot = get_form_required('spot')
        by_support = get_form_required('support')
        by_author = get_form_required('author')
        return redirect(url_for('crunch_data', by_spot=by_spot, by_support=by_support, by_author=by_author))


    return render_template(
        '/crunch_data.html',
        spot_choice = SPOT_CHOICE,
        support_choice=SUPPORT_CHOICE,
        author_choice=['all', g.user.username],
        #author_choice=['all']+[u.username for u in db.session.query(User).all()],
        by_support = by_support,
        by_spot=by_spot,
        by_author=by_author,
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    ).encode(encoding='UTF-8')

if __name__ == "__main__":
    # ***** start app server *******
    app.run(debug=True, host="0.0.0.0", port=9999)
