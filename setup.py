#! /usr/bin/env python3
# This file is part of Dataloop

"""
    After creating setup.py file run the following commands: (delete the build dir)
    bumpversion patch --allow-dirty
    python setup.py bdist_wheel
"""

from setuptools import setup, find_packages

with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

packages = [
    package for package in find_packages() if package.startswith('dtlpymetrics')
]

setup(name='dtlpymetrics',
      classifiers=[
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      version='1.2.10',
      description='Scoring and metrics app',
      author='Dataloop Team',
      author_email='yaya.t@dataloop.ai',
      long_description=readme,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      setup_requires=['wheel'],
      install_requires=requirements,
      include_package_data=True
      )