#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as base
# TODO (mihail): Try a mult-stage Docker file
FROM python:3.8

# TODO (mihail): Don't dump entire directory into image
ADD . /home/fake-news

WORKDIR /home/fake-news

RUN pip install --no-cache-dir -r requirements.txt
