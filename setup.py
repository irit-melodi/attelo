from setuptools import setup, find_packages
setup(
      name     = "attelo",
      version  = "0.2",
      author   = "Philippe Muller, Stergos Afantenos, Pascal Denis",
      author_email = "Philippe.Muller@irit.fr",
      packages = find_packages(exclude=["scripts","experiments","tests"]),
      scripts  = [ "scripts/attelo" ],
      install_requires=['depparse',
                        'enum34',
                        'nltk',
                        'numpy',
                        'Orange',
                        'scipy',
                        'tabulate'],
)
