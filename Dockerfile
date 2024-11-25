FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

RUN pip install --user \
    shapely==2.0.0 \
    seaborn \
    dtlpy

COPY . /app/pkgs/dtlpy-metrics
WORKDIR /app/pkgs/dtlpy-metrics
RUN python /app/pkgs/dtlpy-metrics/setup.py install

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 bash