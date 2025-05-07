# Clean up any existing installation
pip uninstall -y dtlpymetrics

# Change to the package directory
cd /tmp/app/pkgs/dtlpy-metrics

# Build the wheel
python setup.py bdist_wheel

# Install the built wheel
pip install dist/*.whl