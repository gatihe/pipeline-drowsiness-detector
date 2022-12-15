#!/bin/bash
docker build . -f pipeline-application.Dockerfile -t pipeline-base:latest
docker build . -f pipeline-application.Dockerfile -t pipeline-application:latest