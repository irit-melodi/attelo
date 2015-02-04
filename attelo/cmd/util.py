'''
Utility functions for command line tools
'''

from attelo.io import load_data_pack


def load_args_data_pack(args):
    '''
    Load data pack specified via command line arguments
    '''
    return load_data_pack(args.edus,
                          args.pairings,
                          args.features,
                          verbose=not args.quiet)
