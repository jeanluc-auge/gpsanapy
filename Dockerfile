FROM python:3.8

ARG UID=1001
ARG GID=1001
ARG APP_USER=link_planner

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir /gpsanapy
RUN mkdir /gpsanapy/csv_results
RUN mkdir /gpsanapy/gpx_file_upload
RUN mkdir -p /gpsanapy/src/core/
RUN mkdir /gpsanapy/config

COPY ./requirements.txt ./requirements.txt
COPY ./requirements_flask.txt ./requirements_flask.txt

COPY src/core/* /gpsanapy/src/core/
COPY config/* /gpsanapy/config/

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
RUN pip install -r ./requirements_flask.txt

RUN addgroup --gid $GID --system  ${APP_USER} &&\
    adduser --uid $UID --ingroup ${APP_USER} --system ${APP_USER}

RUN chown -R ${UID}:${GID} /gpsanapy
USER ${APP_USER}

WORKDIR /gpsanapy/src/core

CMD gunicorn --bind 0.0.0.0:8080 -t 300 --workers=2 flask_restplus_server:app
