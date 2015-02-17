"visualise attelo outputs"

from __future__ import print_function
from itertools import groupby
from os import path as fp
import os

from .util import (get_output_dir, announce_output_dir)
from ..io import (load_edus, load_predictions, load_gold_predictions)
from ..graph import (DEFAULT_TIMEOUT,
                     select_links,
                     write_dot_graph, to_graph)


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
                           nargs=2,
                           help="gold predictions [pairings, "
                           "features (targets only)]")
    input_grp.add_argument("--predictions", metavar="FILE",
                           help="single predictions")

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


def main_for_harness(args):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    output_dir = get_output_dir(args)
    edus = load_edus(args.edus)
    if args.predictions is not None:
        links = load_predictions(args.predictions)
    else:
        # pylint: disable=star-args
        links = load_gold_predictions(*args.gold)
        # pylint: enable=star-args

    for group, subedus_ in groupby(edus, lambda x: x.grouping):
        subedus = list(subedus_)
        sublinks = select_links(subedus, links)
        if not sublinks:  # not in fold
            continue
        graph = to_graph(group, subedus, sublinks, unrelated=args.unrelated)
        ofilename = fp.join(output_dir, group)
        write_dot_graph(ofilename, graph, quiet=args.quiet,
                        timeout=args.graphviz_timeout)
    announce_output_dir(output_dir)


def main(args):
    "subcommand main (invoked from outer script)"

    main_for_harness(args)
