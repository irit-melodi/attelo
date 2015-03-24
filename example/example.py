from __future__ import print_function

from joblib import (Parallel)

from ..args import (add_common_args, add_decoder_args,
                    add_model_read_args,
                    add_fold_choice_args, validate_fold_choice_args,
                    args_to_decoder, args_to_decoding_mode)
from ..io import (load_model, load_fold_dict)
from ..util import Team
from .util import load_args_data_pack
import attelo.harness.decode as hdecode


DATA_DIR = 'example'

def main():
    dpack = load_data_pack(fp.join(DATA_DIR, 'tiny.edus'),
                           fp.join(DATA_DIR, 'tiny.pairings'),
                           fp.join(DATA_DAT, 'tiny.features.sparse'))
