#!/bin/bash
docker container rm -f $(docker container ls -aq)
docker rmi -f pipeline-application
docker build . -f pipeline-application.Dockerfile -t pipeline-application:latest
docker run -p 80:80 -t -i pipeline-application