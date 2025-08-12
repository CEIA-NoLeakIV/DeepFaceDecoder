FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

RUN mkdir /dataset

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./