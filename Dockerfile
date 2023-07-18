FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

RUN pip install --user \
    shapely==2.0.0 \
    seaborn


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/scoring:0.3.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/scoring:0.3.0
