FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.11_opencv
USER root

RUN /usr/local/bin/python -m pip install \
    shapely>=2.0.4 \
    seaborn

USER 1000
ENV HOME=/tmp

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.27.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.27.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.27.0 bash