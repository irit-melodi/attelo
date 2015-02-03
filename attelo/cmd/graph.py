"visualise attelo outputs"

from __future__ import print_function
from os import path as fp
import codecs
import os
import sys

import pydot

from ..edu import FAKE_ROOT_ID
from ..io import load_predictions
from attelo.harness.util import makedirs


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
def input_to_graph(title, edulinks):
    """
    Convert attelo EDU input to a graph (input should be the
    result of :py:method:load_edus)

    :type edulinks: [(EDU, [string])
    """
    graph = pydot.Dot(title, graph_type='digraph')
    graph.add_node(pydot.Node(FAKE_ROOT_ID, label='.'))
    for edu, links in edulinks:
        child = edu.id
        if child == FAKE_ROOT_ID:
            continue
        attrs = {'shape': 'plaintext'}
        if edu.text:
            attrs['label'] = edu.text
        graph.add_node(pydot.Node(child, **attrs))
        for parent in links:
            attrs = {}
            graph.add_edge(pydot.Edge(parent, child, **attrs))
    return graph


def output_to_graph(title, edulinks):
    """
    Convert attelo predictions to a graph.

    Predictions here consist of an EDU followed by a list of
    (parent name, relation label) tuples

    :type edulinks: [(EDU, [(string, string)])
    """
    graph = pydot.Dot(title, graph_type='digraph')
    graph.add_node(pydot.Node(FAKE_ROOT_ID, label='.'))
    for edu, links in edulinks:
        child = edu.id
        if child == FAKE_ROOT_ID:
            continue
        attrs = {'shape': 'plaintext'}
        if edu.text:
            attrs['label'] = edu.text
        graph.add_node(pydot.Node(child, **attrs))
        for parent, label in links:
            attrs = {}
            if label:
                attrs['label'] = label
            graph.add_edge(pydot.Edge(parent, child, **attrs))
    return graph
# pylint: enable=star-args


def config_argparser(psr):
    "add subcommand arguments to subparser"

    psr.add_argument("graph", metavar="FILE",
                     help="attelo output file")
    psr.add_argument("output", metavar="DIR",
                     help="output directory for graphs")
    psr.set_defaults(func=main)


def main_for_harness(args):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    edulinks = load_predictions(args.graph)
    graph_name = fp.basename(args.graph)
    graph = output_to_graph(graph_name, edulinks)
    ofilename = fp.join(args.output, graph_name)
    write_dot_graph(ofilename, graph)


def main(args):
    "subcommand main (invoked from outer script)"

    main_for_harness(args)
