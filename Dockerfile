FROM python:3.12.0-bullseye

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip install -r requirements.txt