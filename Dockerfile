FROM python:3.11

WORKDIR /workspaces/eqmarl
COPY requirements*.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        -r requirements.txt \
        -r requirements-dev.txt