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


def _load_links(args):
    """
    Return edus, source links, and maybe target links
    """
    edus = load_edus(args.edus)
    if args.predictions is not None:
        src_links = load_predictions(args.predictions)
    elif args.gold is not None:
        # pylint: disable=star-args
        src_links = load_gold_predictions(*args.gold)
        # pylint: enable=star-args
    else:
        raise Exception('TODO need arg validation to trap this case')

    if args.diff_to is not None:
        tgt_links = load_predictions(args.diff_to)
    else:
        tgt_links = None

    return edus, src_links, tgt_links


def main_for_harness(args):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    output_dir = get_output_dir(args)
    edus, links, tgt_links = _load_links(args)
    for group, subedus_ in groupby(edus, lambda x: x.grouping):
        if args.select is not None and group not in args.select:
            continue
        subedus = list(subedus_)
        sublinks = select_links(subedus, links,
                                intra=args.intra,
                                inter=args.inter)
        if not sublinks:  # not in fold
            continue
        # skip any groups that are not in diff target (envisioned
        # use case, diffing gold against an output)
        if tgt_links is None:
            tgt_sublinks = None
        else:
            tgt_sublinks = select_links(subedus, tgt_links,
                                        intra=args.intra,
                                        inter=args.inter)
            if not tgt_sublinks:
                continue
        graph = to_graph(group, subedus, sublinks,
                         tgt_links=tgt_sublinks,
                         unrelated=args.unrelated,
                         inter=args.inter)
        ofilename = fp.join(output_dir, group)
        write_dot_graph(ofilename, graph, quiet=args.quiet,
                        timeout=args.graphviz_timeout)
    announce_output_dir(output_dir)


def main(args):
    "subcommand main (invoked from outer script)"

    main_for_harness(args)
