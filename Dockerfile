FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv
USER root
COPY . /tmp/app/pkgs/dtlpy-metrics
RUN chmod -R 777 /tmp/app && chown -R 1000:1000 /tmp/app

RUN /usr/local/bin/python -m pip install \
    shapely==2.0.0 \
    seaborn \
    dtlpy
RUN pip uninstall -y dtlpymetrics
RUN cd /tmp/app/pkgs/dtlpy-metrics && /usr/local/bin/python setup.py install

USER 1000
ENV HOME=/tmp
# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.19.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.19.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.19.0 bash