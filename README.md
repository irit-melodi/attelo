These are experiments being conducted around 2014-07 by the
Melodi team in IRIT on the RST Discourse Treebank corpus.

## Prerequisites

1. Python 2.7. Python 3 might also work.
2. pip (see educe README if you do not)
3. git (to keep up with educe/attelo changes)
4. graphviz (for visualising graphs) (TODO: check if needed)
5. [optional] Anacoda (`conda --version` should say 2.2.6 or higher)
6. a copy of the RST Discourse Treebank

## Sandboxing

If you are attempting to use the development version of this code
(ie. from SVN), I highly recommend using a sandbox environment.
We have two versions below, one for Anaconda users (on Mac),
and one for standard Python users via virtualenv.

### For Anaconda users

Anaconda users get slightly different instructions because virtualenv
doesn't yet seem to work well with it (at least with the versions we've
tried). Instead of using virtualenv, you could try something like this

    conda create -n irit-rst-dt --clone $HOME/anaconda

If that doesn't work, make sure your anaconda version is up to date,
and try `/anaconda` instead of `$HOME/anaconda`.

Note that whenever you want to use STAC things, you would need to run
this command

    source activate irit-rst-dt

### For standard Python users

The virtualenv equivalent works a bit more like the follow:

    mkdir $HOME/.virtualenvs
    virtualenv $HOME/.virtualenvs/irit-rst-dt --no-site-packages

Whenever you want to use STAC things, you would need to run this
command

    source $HOME/.virtualenvs/irit-rst-dt/bin/activate

## Installation (development mode)

1. Activate your virtual environment (see above)

2. Install this package and its dependencies

       pip install -r requirements.txt

3. Link your copy of the RST DT corpus in, for example:

       ln -s $HOME/CORPORA/rst_discourse_treebank/data corpus

4. Install megam (for attelo).
   This might be tricky if you're on a Mac.
   Ask Eric.

## Usage

Running the pieces of infrastructure here should consist of running
`irit-rst-dt <subcommand>`

* `irit-rst-dt gather`: extract features
* `irit-rst-dt evaluate`: run n-fold attachment/labelling experiment (slow)
