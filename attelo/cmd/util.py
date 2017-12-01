'''
Utility functions for command line tools
'''

from __future__ import print_function
from os import path as fp
import os
import sys
import tempfile

from attelo.io import load_multipack


def load_args_multipack(args):
    '''
    Load multipack specified via command line arguments
    '''
    return load_multipack(args.edus, args.pairings,
                          args.features,
                          args.vocab, args.labels,
                          file_split='corpus',  # WIP
                          verbose=not args.quiet)


def get_output_dir(args):
    """
    Return the output directory specified on (or inferred from) the command
    line arguments, *creating it if necessary*.

    We try the following in order:

    1. If `--output` is given explicitly, we'll just use/create that
    2. Otherwise, just make a temporary directory. Later on, you'll probably
    want to call `announce_output_dir`.
    """
    if args.output:
        if os.path.isfile(args.output):
            oops = "Sorry, {} already exists and is not a directory"
            sys.exit(oops.format(args.output))
        elif not fp.isdir(args.output):
            os.makedirs(args.output)
        return args.output
    else:
        return tempfile.mkdtemp()


def announce_output_dir(output_dir):
    """
    Tell the user where we saved the output
    """
    print("Output files written to", output_dir, file=sys.stderr)
