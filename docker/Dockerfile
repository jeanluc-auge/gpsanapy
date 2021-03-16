#   Dockerfile for gpsanapy with the 'recursive-directory' option
#   executes all gpx files in /csv_results shared directory
#   and returns the csv results in the same shared directory
#   BUILD & RUN with :
# 	sudo docker build --no-cache -t gpsanapy .
# 	sudo docker run -it --rm -v /archives:/gpsanapy/gpx_files -v /logs:/gpsanapy/csv_results -v /config:/gpsanapy/config gpsanapy

FROM python:3.7
RUN pip3 install --upgrade pip
RUN pip3 install -r /gpsanapy/docker_requirements.txt
RUN git clone https://github.com/jeanluc-auge/gpsanapy.git
RUN mkdir /gpsanapy/csv_results
RUN mkdir /gpsanapy/gpx_files
WORKDIR /gpsanapy/
CMD python3 src/core/gps_analysis.py -d gpx_files # use -rd for recursive directory search