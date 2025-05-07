pip uninstall -y dtlpymetrics

# cd /tmp/app/pkgs/dtlpy-metrics && /usr/local/bin/python setup.py install
# Change to the package directory
cd /tmp/app/pkgs/dtlpy-metrics

# Build the wheel
python setup.py bdist_wheel

# Install the built wheel
pip install dist/*.whl
