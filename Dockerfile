FROM gcr.io/viewo-g/piper/agent/runner/cpu/main:1.80.6.latest

RUN pip3 install --user shapely>=2.0.0


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/scoring:0.1.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/scoring:0.1.0
