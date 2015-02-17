"graph visualisation"

from __future__ import print_function
from os import path as fp
import codecs
import signal
import subprocess
import sys

import pydot

from .edu import FAKE_ROOT_ID
from .harness.util import makedirs
from .table import UNRELATED


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
