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
#from flask_mail import Mail


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

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../../")
ALLOWED_EXTENSIONS = {'.gpx', '.sml'}
SUPPORT_CHOICE = ['all', 'windsurf', 'windfoil', 'kitesurf', 'kitefoil', 'kayak']
SPOT_CHOICE = ['all', 'g13', 'nantouar', 'trestel', 'keriec', 'bnig', 'other']
# check or create that file upload and database dir exist:
dir_path = [os.path.join(ROOT_DIR, d) for d in ('database', 'gpx_file_upload')]
for d in dir_path:
    if not Path(d).is_dir():
        os.makedirs(d)
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'gpx_file_upload')
DATABASE_PATH = os.path.join(ROOT_DIR, 'database/test.db')
database = Path(DATABASE_PATH).resolve()

# ******* define Flask api *******
app = Flask(__name__, instance_relative_config=True, static_url_path='/static', static_folder='static')

# *** sql ****
# engine = create_engine('sqlite:///:memory:', echo=True)
# Base = declarative_base()
app.config.from_mapping(SECRET_KEY='dev')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{database}"
#'sqlite:////home/jla/gps/database/test.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# # TODO gmail account & configuration
# app.config['MAIL_SERVER'] = 'smtp.example.com'
# app.config['MAIL_PORT'] = 465
# app.config['MAIL_USE_SSL'] = False
# app.config['MAIL_USE_TLS'] = Truepip
# app.config['MAIL_USERNAME'] = 'username'
# app.config['MAIL_PASSWORD'] = 'password'
# mail = Mail(app)
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


# *************** data model *******************

def check_df(fn):
    """
    on the fly wrapper (decorator)
    to catch errors and check validity of returned df
    """
    @functools.wraps(fn)
    def wrapped_fn(**kwargs):
        try:
            df = fn(**kwargs)
        except Exception as e:
            message = (
                f"an unexpected error {e} occured\n"
                f"when running {fn.__name__}\n"
                f"with args {kwargs}\n"
            )
            logger.warning(message)
            flash(message)
            return None

        if df is None or df.empty:
            message = (
                f"there is not enough data in {kwargs}"
            )
            logger.warning(message)
            flash(message)
            return None

        return df
    return wrapped_fn

def run_file(gpsana_client, file):
    """
    run TraceAnalysis gpsana_client and check status
    :param gpsana_client: TraceAnalysis instance
    :param file: dB file
    :return:
    """
    status, error = gpsana_client.run()
    error = '\n'.join(error)
    if not status:
        if not file.approved:
            message = f"unapproved file {file.filename} with error {error} will be removed"
            del_file(file)
        else:
            message = f"analysis {file.filename} yields error {error}"
        flash(message)
        logger.warning(message)
        flash('\n'.join(error))
        return redirect(url_for('files'))

def get_file_attr(filename):
    """
    return file status, file stem, file path and check it is valid
    :param filename: uploaded filename
    :return: (filename wo extension, full file path)
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    filename, file_extension = split_path(filename)
    logger.info(f"uploading file {filename} with extension {file_extension} and path {file_path}")
    if db.session.query(TraceFiles).filter(TraceFiles.filename == filename).first() is not None:
        flash(f"file {filename} already exists")
        return redirect(request.url)
    if file_extension not in ALLOWED_EXTENSIONS:
        flash(f"file extension {file_extension} is not supported")
        return redirect(request.url)
    return filename, file_path
    # return '.' in filename and \
    #        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def del_file(file):
    """
    delete file in database, disk and ranking results
    :param file: dB file to remove
    :return:
    """
    try:
        db.session.delete(file)
        db.session.commit()
        trace.delete_result(file.filename, file.file_path)
    except Exception as e:
        logger.warning(f"an error occured when removing file {file.filename}: {e}")

def get_file(id, check_author=True, check_admin=False):
    """
    database query of file id
    optionally check if the session user is allowed to query the file
    :param id: int file id in database
    :param check_author: bool check if session user matches the file author
    :return: database file
    """
    file = db.session.query(TraceFiles).filter(TraceFiles.id == id).first()
    if file is None:
        abort(404, f"file id {id} doesn't exist.")
    if check_author and file.user_id != g.user.id and g.user.username!='admin':
        abort(403)
    if check_admin and g.user.username!='admin':
        abort(403)

    return file

def post_file(file, filename_with_ext, **kwargs):
    """
    new file is saved to disk and put in database
    :param file: request.files object
    :param filename_with_ext: str full filename
    :param kwargs:
        str spot
        str support
    :return: database file object
    """
    filename, file_path = get_file_attr(filename_with_ext)
    file.save(file_path)
    new_file = TraceFiles(
        filename=filename,
        file_path=file_path,
        spot=kwargs.get('spot', ''),
        support=kwargs.get('support', ''),
        user_id=g.user.id,
        approved=False,
    )
    db.session.add(new_file)
    db.session.commit()
    return new_file

def get_form_required(item_name):
    item = request.form[item_name]
    if not item:
        flash(f'No {item_name} part')
        return redirect(request.url)
    return item

def gen_bokeh_resources(p):
    """
    generate bokeh html ressources to render graphs
    :param p: bokeh plot object
    :return: dict of bokeh html ressources to render in html template
    """
    try:
        bokeh_resources = {}
        bokeh_resources['js_resources'] = INLINE.render_js()
        bokeh_resources['css_resources'] = INLINE.render_css()
        bokeh_resources['script'], bokeh_resources['div'] = components(p)
    except Exception:
        flash("cannot display graph")
        return redirect(request.url)
    return bokeh_resources

# ********* gps flask server *************
@app.before_request
def load_logged_in_user():
    """load session user id in flask g"""
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = db.session.query(User).filter(User.id==user_id).first()

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = get_form_required('username')
        password = get_form_required('password')
        email = get_form_required('email')
        location = request.form['location']
        infos = request.form['infos']

        error = None
        if db.session.query(User).filter(User.username==username).first() is not None:
            error = f'User {username} is already registered.'
        elif db.session.query(User).filter(User.email==email).first() is not None:
            error = f'User {email} is already registered.'

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
        username = get_form_required('username')
        password = get_form_required('password')

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files', methods=('GET', 'POST'))
@app.route('/files/<string:by_support>/<string:by_spot>/<string:by_author>/<string:file_status>', methods=('GET', 'POST'))
def files(by_support='all', by_spot='all', by_author='all', file_status='all'):

    if request.method == 'POST':
        by_spot = request.form['spot']
        by_support = request.form['support']
        by_author = request.form.get('author', 'all')
        file_status = request.form.get('file_status','all')
        return redirect(url_for(
            'files',
            by_spot=by_spot,
            by_support=by_support,
            by_author=by_author,
            file_status=file_status,
        ))
    files = db.session.query(TraceFiles)
    if by_spot != 'all':
        files = files.filter_by(spot=by_spot)
    elif by_support != 'all':
        files = files.filter_by(support=by_support)
    elif by_author != 'all':
        user = db.session.query(User).filter(User.username == by_author).first()
        files = files.filter_by(user=user)
    elif file_status != 'all':
        files = files.filter_by(approved=False)
    files = files.order_by(TraceFiles.id.desc()).all()
    if g.user:
        author_choice = ['all', g.user.username]
    else:
        author_choice = ['all']

    return render_template(
        'files.html',
        files=files,
        spot_choice=SPOT_CHOICE,
        support_choice=SUPPORT_CHOICE,
        author_choice=author_choice,
        status_choice=['all', 'not approved'],
        # author_choice=['all']+[u.username for u in db.session.query(User).all()],
        by_support=by_support,
        by_spot=by_spot,
        by_author=by_author,
    )

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

        new_file = post_file(file, filename, spot=spot, support=support)

        return redirect(url_for('analyse', id=new_file.id))
    return render_template('upload_file.html', support_choice=SUPPORT_CHOICE, spot_choice=SPOT_CHOICE)

@app.route('/<int:id>/delete_file', methods=('GET', 'POST'))
@login_required
def delete_file(id):
    file = get_file(id, check_author=True) # check it exists
    del_file(file)
    return redirect(url_for('files'))

@app.route('/<int:id>/approve_file', methods=('GET', 'POST'))
@login_required
def approve_file(id):
    file = get_file(id, check_admin=True) # check it exists
    #file.update({'approved': True})
    file.approved=True
    db.session.add(file)
    db.session.commit()

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
    gpsana_client = TraceAnalysis(file.file_path, author=file.user.username, spot=file.spot, support=file.support)
    run_file(gpsana_client, file)
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
    reduced_results = check_df(trace.reduced_results)(
            #by_support=file.support,
            author=file.user.username,
        )
    if reduced_results is not None:
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
    tables = []
    titles = None
    df = check_df(trace.rank_all_results)(support=support)
    if df is not None:
        tables = [df.to_html(classes='data')]
        titles = df.columns.values

    if request.method == 'POST':
        support = get_form_required('support')
        return redirect(url_for('ranking', support=support))

    return render_template(
        '/ranking.html',
        support_choice=SUPPORT_CHOICE,
        tables = tables,
        titles = titles,
        support = support
    )

@app.route('/crunch_data', methods=('GET', 'POST'))
@app.route('/crunch_data/<string:by_support>/<string:by_spot>/<string:by_author>',  methods=('GET', 'POST'))
def crunch_data(by_support='all', by_spot='all', by_author='all'):
    reduced_results = check_df(trace.reduced_results)(
            support=by_support,
            spot=by_spot,
            author=by_author,
            check_config=True
        )
    bokeh_template = {}
    if reduced_results is not None:
        p = all_results_speed_density(reduced_results)
        # grab the static resources
        bokeh_template['plot speed data'] = gen_bokeh_resources(p)

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
        bokeh_template=bokeh_template,
    ).encode(encoding='UTF-8')

if __name__ == "__main__":
    # ***** start app server *******
    app.run(debug=True, host="0.0.0.0", port=9999)
