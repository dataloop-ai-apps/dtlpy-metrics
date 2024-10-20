FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

RUN pip install --user \
    shapely==2.0.0 \
    seaborn \
    dtlpy \
    git+https://github.com/dataloop-ai-apps/dtlpy-metrics@CE-639_set_threshold

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/dtlpy-metrics:0.4.0 bash