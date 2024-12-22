FROM dataloopai/dtlpy-agent:cpu.py3.10.opencv
USER root
COPY . /tmp/app/pkgs/dtlpy-metrics
RUN chmod -R 777 /tmp/app && chown -R 1000:1000 /tmp/app

USER 1000
ENV HOME=/tmp
ENV PATH=$HOME/.local/bin:/usr/local/bin:$PATH
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN /usr/local/bin/python -m pip install --user \
    shapely==2.0.0 \
    seaborn \
    dtlpy

RUN cd /tmp/app/pkgs/dtlpy-metrics && /usr/local/bin/python setup.py install --prefix=$HOME/.local


# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.11.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.11.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.11.0 bash