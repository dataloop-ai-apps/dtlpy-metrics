FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv

USER 1000
RUN pip install --user \
    shapely==2.0.0 \
    seaborn \
    dtlpy

COPY . /tmp/app/pkgs/dtlpy-metrics
USER root
WORKDIR /tmp/app/pkgs/dtlpy-metrics
RUN python setup.py install
USER 1000

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 bash