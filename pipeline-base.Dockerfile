#Docker image with steps to setup large configs for image proccessing tools such as 
#gcc, cmake and dlib

# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install build-essential -y
RUN apt-get -y install cmake
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /

# Install pip requirements
RUN python3 -m pip install --upgrade pip setuptools wheel                                                                                                                                                                                                
RUN python3 -m pip install dlib
RUN python3 -m pip install cmake
RUN python3 -m pip install imutils
RUN python3 -m pip install requests
RUN python3 -m pip install -r requirements.txt 