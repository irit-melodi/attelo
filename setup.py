from setuptools import setup
import glob
import os

setup(name='irit-rst-dt',
      version='0.1',
      author='Eric Kow',
      author_email='eric@erickow.com',
      packages=[],
      scripts=[f for f in glob.glob('scripts/*') if not os.path.isdir(f)],
      requires=['educe', 'attelo']
      )
