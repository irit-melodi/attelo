'''
attelo installation
'''

from setuptools import setup, find_packages

setup(name="attelo",
      version="0.3",
      author="IRIT Melodi team",
      author_email="Philippe.Muller@irit.fr",
      packages=find_packages(exclude=["scripts",
                                      "experiments",
                                      "tests"]),
      scripts=["scripts/attelo"],
      install_requires=['depparse',
                        'enum34',
                        'joblib',
                        'mock',
                        'nltk',
                        'numpy',
                        'scikit-learn',
                        'six',
                        'scipy >= 0.14.0',
                        'tabulate'])
