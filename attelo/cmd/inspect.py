"show properties about models"

from __future__ import print_function
import codecs

from ..args import (add_model_read_args)
from ..io import (load_labels, load_model, load_vocab)
from ..score import (discriminating_features)
from ..report import (show_discriminating_features)
from ..table import UNKNOWN
from ..util import (Team)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


DEFAULT_TOP = 3
'default top number of features to show'


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_model_read_args(psr, "{} model to inspect")
    psr.add_argument("features", metavar="FILE",
                     help="sparse features file (just for labels)")
    psr.add_argument("vocab", metavar="FILE",
                     help="feature vocabulary")
    psr.add_argument("--top", metavar="N", type=int,
                     default=DEFAULT_TOP,
                     help=("show the best N features "
                           "(default: {})".format(DEFAULT_TOP)))
    psr.add_argument("--output", metavar="FILE",
                     help="output to file")
    psr.set_defaults(func=main)


def main(args):
    "subcommand main (invoked from outer script)"
    models = Team(attach=load_model(args.attachment_model),
                  label=load_model(args.relation_model))
    # FIXME find a clean way to properly read ready-for-use labels
    # upstream ; true labels are 1-based in svmlight format but 0-based
    # for sklearn
    labels = [UNKNOWN] + load_labels(args.features)
    vocab = load_vocab(args.vocab)
    discr = discriminating_features(models, labels, vocab, args.top)
    res = show_discriminating_features(discr)
    if args.output is None:
        print(res)
    else:
        with codecs.open(args.output, 'wb', 'utf-8') as fout:
            print(res, file=fout)
