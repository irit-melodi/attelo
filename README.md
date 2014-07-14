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

3. Install megam (for attelo).
   This might be tricky if you're on a Mac.
   Ask Eric.

## Parser infrastructure

Now that you have everything installed, there are a handful of parser
infrastructure scripts which run the feature extraction process, build
the attachment/labeling models, and run the decoder on sample data.

code/parser/gather-features
~ do feature extraction from annotated data, along with pre-saved
  pos-tagging and and parser output, and some lexical resources

code/parser/build-model
~ from the extracted features (see gather-features), build the
  attachment and labeling models needed to run the parser

code/parser/stac-parser.sh
~ given a model (see build-model), and a STAC soclog file, run the
  parser and display a graph of the output (needs third party
  tools, see script for details)

code/parser/harness.sh
~ given extracted features (see gather-features), run experiments on
  STAC data
