"visualise attelo outputs"

from __future__ import print_function
from itertools import groupby
from os import path as fp
import codecs
import os
import signal
import subprocess
import sys

import pydot

from .util import (get_output_dir, announce_output_dir)
from ..edu import FAKE_ROOT_ID
from ..io import (load_edus, load_predictions, load_gold_predictions)
from ..table import UNRELATED
from attelo.harness.util import makedirs


DEFAULT_TIMEOUT = 30


class Alarm(Exception):
    "Exception to raise on signal timeout"
    pass


# pylint: disable=unused-argument
def alarm_handler(_, frame):
    "Raise Alarm on signal"
    raise Alarm
# pylint: enable=unused-argument


def select_links(edus, links):
    """
    Given a set of edus and of edu id pairs, return only the pairs
    whose ids appear in the edu list
    """
    edu_ids = frozenset(edu.id for edu in edus)
    return [(e1, e2, l) for e1, e2, l in links
            if e1 in edu_ids or e2 in edu_ids]


def mk_parent_dirs(filename):
    """
    Given a filepath that we want to write, create its parent directory as
    needed.
    """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def write_dot_graph(filename, dot_graph,
                    run_graphviz=True,
                    quiet=False,
                    timeout=DEFAULT_TIMEOUT):
    """
    Write a dot graph and possibly run graphviz on it
    """
    makedirs(fp.dirname(filename))
    dot_file = filename + '.dot'
    svg_file = filename + '.svg'
    with codecs.open(dot_file, 'w', encoding='utf-8') as dotf:
        print(dot_graph.to_string(), file=dotf)
    if run_graphviz:
        if not quiet:
            print("Creating %s" % svg_file, file=sys.stderr)
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)  # half a minute
        try:
            subprocess.call(["dot",
                             "-T", "svg",
                             "-o", svg_file,
                             dot_file])
            signal.alarm(0)  # reset the alarm
        except Alarm:
            print("Killed graphviz because it was taking too long",
                  file=sys.stderr)


# pylint: disable=star-args
def _build_core_graph(title, edus):
    """
    Return a graph containing just nodes
    """
    graph = pydot.Dot(title, graph_type='digraph')
    graph.add_node(pydot.Node(FAKE_ROOT_ID, label='.'))
    groups = sorted(set(e.subgrouping for e in edus))
    subgraphs = {}
    for grp in groups:
        attrs = {'color': 'lightgrey',
                 'label': grp,
                 'style': 'dashed'}
        subgraphs[grp] = pydot.Subgraph('cluster_' + grp, **attrs)
        graph.add_subgraph(subgraphs[grp])

    for edu in edus:
        if edu.id == FAKE_ROOT_ID:
            continue
        attrs = {'shape': 'plaintext'}
        if edu.text:
            # we add a space to force pydot to quote this
            # (its need-to-quote detector isn't always reliable)
            attrs['label'] = edu.text + ' '
        subgraphs[edu.subgrouping].add_node(pydot.Node(edu.id, **attrs))
    return graph


def to_graph(title, edus, links, unrelated=False):
    """
    Convert attelo predictions to a graph.

    Predictions here consist of an EDU followed by a list of
    (parent name, relation label) tuples

    :type edulinks: [(EDU, [(string, string)])
    """
    graph = _build_core_graph(title, edus)
    for parent, child, label in links:
        attrs = {}
        if label != UNRELATED:
            attrs['label'] = label
            graph.add_edge(pydot.Edge(parent, child, **attrs))
        elif unrelated:
            attrs = {'style': 'dashed',
                     'color': 'grey'}
            graph.add_edge(pydot.Edge(parent, child, **attrs))
    return graph
# pylint: enable=star-args


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
