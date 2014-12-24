'''
irit-rst-dt setup: irit-rst-dt is an experiment harness.

Its job is to reduce complex experiment pipelines to simple
commands like `irit-rst-dt gather` and `irit-rst-dt evaluate`
'''

from setuptools import setup, find_packages
import glob
import os

setup(name='irit-rst-dt',
      version='0.1',
      author='Eric Kow',
      author_email='eric@erickow.com',
      packages=find_packages(),
      scripts=[f for f in glob.glob('scripts/*') if not os.path.isdir(f)],
      install_requires=['educe', 'attelo'])
