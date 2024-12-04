FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

USER root

ENV HOME=/tmp
USER 1000

RUN pip install --user \
    shapely==2.0.0 \
    seaborn \
    dtlpy

RUN mkdir /tmp/app/pkgs/dtlpy-metrics
COPY . /tmp/app/pkgs/dtlpy-metrics
WORKDIR /tmp/app/pkgs/dtlpy-metrics
RUN python setup.py install

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 bash