"visualise attelo outputs"

from __future__ import print_function
from itertools import groupby
from os import path as fp
import codecs
import os
import sys
import tempfile

import pydot

from ..edu import FAKE_ROOT_ID
from ..io import load_edus, load_predictions
from ..table import UNRELATED
from attelo.harness.util import makedirs


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


def write_dot_graph(filename, dot_graph, run_graphviz=True):
    """
    Write a dot graph and possibly run graphviz on it
    """
    makedirs(fp.dirname(filename))
    dot_file = filename + '.dot'
    svg_file = filename + '.svg'
    with codecs.open(dot_file, 'w', encoding='utf-8') as dotf:
        print(dot_graph.to_string(), file=dotf)
    if run_graphviz:
        print("Creating %s" % svg_file, file=sys.stderr)
        os.system('dot -T svg -o %s %s' % (svg_file, dot_file))


# pylint: disable=star-args
def _build_core_graph(title, edus):
    """
    Return a graph containing just nodes
    """
    graph = pydot.Dot(title, graph_type='digraph')
    graph.add_node(pydot.Node(FAKE_ROOT_ID, label='.'))
    for edu in edus:
        if edu.id == FAKE_ROOT_ID:
            continue
        attrs = {'shape': 'plaintext'}
        if edu.text:
            attrs['label'] = edu.text
        graph.add_node(pydot.Node(edu.id, **attrs))
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


def config_argparser(psr):
    "add subcommand arguments to subparser"

    psr.add_argument("edus", metavar="FILE",
                     help="attelo edu input file")
    psr.add_argument("predictions", metavar="FILE",
                     help="attelo predictions file")
    psr.add_argument("--output", metavar="DIR",
                     help="output directory for graphs")
    psr.add_argument("--unrelated",
                     action='store_true',
                     help="include unrelated pairs")
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
    links = load_predictions(args.predictions)
    for group, subedus_ in groupby(edus, lambda x: x.grouping):
        subedus = list(subedus_)
        sublinks = select_links(subedus, links)
        if not sublinks:  # not in fold
            continue
        graph = to_graph(group, subedus, sublinks, unrelated=args.unrelated)
        ofilename = fp.join(output_dir, group)
        write_dot_graph(ofilename, graph)
    announce_output_dir(output_dir)


def main(args):
    "subcommand main (invoked from outer script)"

    main_for_harness(args)
