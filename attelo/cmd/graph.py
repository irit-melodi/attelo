"visualise attelo outputs"

from __future__ import print_function
import os

from .util import (get_output_dir, announce_output_dir)
from ..io import (load_edus, load_predictions, load_gold_predictions)
from ..graph import (DEFAULT_TIMEOUT,
                     GraphSettings,
                     diff_all,
                     graph_all)


def mk_parent_dirs(filename):
    """
    Given a filepath that we want to write, create its parent directory as
    needed.
    """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def config_argparser(psr):
    "add subcommand arguments to subparser"

    psr.add_argument("edus", metavar="FILE",
                     help="attelo edu input file")

    input_grp = psr.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--gold", metavar="FILE",
                           nargs=3,
                           help="gold predictions [pairings, "
                           "features (targets only), labels]")
    input_grp.add_argument("--predictions", metavar="FILE",
                           help="single predictions")

    psr.add_argument("--diff-to", metavar="FILE",
                     help="single predictions to diff against [target]")

    sent_grp = psr.add_mutually_exclusive_group()
    sent_grp.add_argument("--intra",
                          action='store_true',
                          default=False,
                          help="hide links between subgroupings")
    sent_grp.add_argument("--inter",
                          action='store_true',
                          default=False,
                          help="hide links within subgroupings")

    psr.add_argument("--select", nargs='*',
                     metavar='GROUPING',
                     help="only show graphs for these groupings")

    psr.add_argument("--output", metavar="DIR",
                     help="output directory for graphs")
    psr.add_argument("--quiet", action="store_true",
                     help="Supress all feedback")
    psr.add_argument("--unrelated",
                     action='store_true',
                     help="include unrelated pairs")
    psr.add_argument("--graphviz-timeout", metavar="N",
                     type=int,
                     default=DEFAULT_TIMEOUT,
                     help=("seconds before timing graphviz out "
                           "(default {})").format(DEFAULT_TIMEOUT))
    psr.set_defaults(func=main)


def main(args):
    "subcommand main (invoked from outer script)"
    if args.intra:
        to_hide = 'inter'
    elif args.inter:
        to_hide = 'intra'
    else:
        to_hide = None
    settings = GraphSettings(hide=to_hide,
                             select=args.select,
                             unrelated=args.unrelated,
                             timeout=args.graphviz_timeout,
                             quiet=args.quiet)
    output_dir = get_output_dir(args)

    edus = load_edus(args.edus)
    if args.predictions is not None:
        src_links = load_predictions(args.predictions)
    elif args.gold is not None:
        # pylint: disable=star-args
        src_links = load_gold_predictions(*args.gold)
        # pylint: enable=star-args
    else:
        raise Exception('TODO need arg validation to trap this case')

    if args.diff_to is None:
        graph_all(edus, src_links, settings, output_dir)
    else:
        tgt_links = load_predictions(args.diff_to)
        diff_all(edus, src_links, tgt_links, settings, output_dir)

    announce_output_dir(output_dir)
