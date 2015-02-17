"graph visualisation"

from __future__ import print_function
from collections import defaultdict
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


def select_links(edus, links, intra=False, inter=False):
    """
    Given a set of edus and of edu id pairs, return only the pairs
    whose ids appear in the edu list

    :param intra: if True, in addition to the constraints above,
                  only return links that are in the same subgrouping
    :param inter: if True, only return links between subgroupings
    """
    subgroupings = {edu.id: edu.subgrouping for edu in edus}
    edu_ids = subgroupings.keys()
    if inter:
        slinks = [(subgroupings.get(e1, FAKE_ROOT_ID),
                   subgroupings.get(e2, FAKE_ROOT_ID), l)
                  for e1, e2, l in links]
        return [(s1, s2, l) for (s1, s2, l) in slinks if s1 != s2]
    else:
        return [(e1, e2, l) for e1, e2, l in links
                if (e1 in edu_ids or e2 in edu_ids)
                and (not intra or subgroupings.get(e1) == subgroupings.get(e2))]


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
def _build_core_graph(title, edus, inter=False):
    """
    Return a graph containing just nodes

    If inter is true, we only create nodes for whole sentences.
    We build these by concatenating entire edus in the
    same subgrouping
    """
    graph = pydot.Dot(title, graph_type='digraph')
    graph.add_node(pydot.Node(FAKE_ROOT_ID, label='.'))
    groups = defaultdict(list)
    for edu in sorted(edus, key=lambda e: e.span):
        groups[edu.subgrouping].append(edu)

    subgraphs = {}
    for grp in groups:
        attrs = {'color': 'lightgrey',
                 'label': grp,
                 'style': 'dashed'}
        subgraphs[grp] = pydot.Subgraph('cluster_' + grp, **attrs)
        graph.add_subgraph(subgraphs[grp])
        if inter:
            attrs = {'shape': 'plaintext'}
            mega_text = '\n'.join(e.text for e in
                                  sorted(groups[grp],
                                         key=lambda x: x.span())
                                  if e.text is not None)
            if mega_text:
                attrs['label'] = mega_text + ' '
            subgraphs[grp].add_node(pydot.Node(grp, **attrs))

    for edu in edus:
        if inter:
            break
        if edu.id == FAKE_ROOT_ID:
            continue
        attrs = {'shape': 'plaintext'}
        if edu.text:
            # we add a space to force pydot to quote this
            # (its need-to-quote detector isn't always reliable)
            attrs['label'] = edu.text + ' '
        subgraphs[edu.subgrouping].add_node(pydot.Node(edu.id, **attrs))
    return graph


def _diff_links(src_links, tgt_links):
    """
    Return

    * both: links in both sets
    * src_only: links in src only
    * tgt_only: links in tgt only
    * neither: UNRELATED in both sets
    """

    src_dict = {(p, c): l for p, c, l in src_links
                if l != UNRELATED}
    tgt_dict = {(p, c): l for p, c, l in tgt_links
                if l != UNRELATED}
    neither =\
        frozenset(pcl for pcl in src_links if pcl[2] == UNRELATED) |\
        frozenset(pcl for pcl in tgt_links if pcl[2] == UNRELATED)

    both = []
    src_only = []
    tgt_only = []

    for key in frozenset(src_dict) | frozenset(tgt_dict):
        parent, child = key
        src_label = src_dict.get(key)
        tgt_label = tgt_dict.get(key)
        if src_label is not None and tgt_label is not None:
            label = src_label if src_label == tgt_label else\
                "{} | {}".format(src_label, tgt_label)
            both.append((parent, child, label))
        elif src_label is not None:
            src_only.append((parent, child, src_label))
        elif tgt_label is not None:
            tgt_only.append((parent, child, tgt_label))

    return both, src_only, tgt_only, neither


def to_graph(title, edus, links,
             unrelated=False,
             tgt_links=None,
             inter=False):
    """
    Convert attelo predictions to a graph.

    Predictions here consist of an EDU followed by a list of
    (parent name, relation label) tuples

    :param tgt_links: if present, we generate a graph that
                      represents a difference between the
                      links and tgt_links (by highlighting
                      links that only occur in one or the
                      other)

    :type edulinks: [(EDU, [(string, string)])
    """
    graph = _build_core_graph(title, edus, inter=inter)
    both, src_only, tgt_only, neither = _diff_links(links, tgt_links or [])
    if tgt_links is None:
        both = src_only
        src_only = []

    # both - standard
    for parent, child, label in both:
        attrs = {'label': label}
        graph.add_edge(pydot.Edge(parent, child, **attrs))

    # src_only - indicate excess (thick RED)
    for parent, child, label in src_only:
        attrs = {'label': label,
                 'penwidth': 2,
                 'color': 'red'}
        graph.add_edge(pydot.Edge(parent, child, **attrs))

    # tgt_only - indicate missing (thick grey)
    for parent, child, label in tgt_only:
        attrs = {'label': label,
                 'penwidth': 2,
                 'style': 'dashed',
                 'color': 'blue'}
        graph.add_edge(pydot.Edge(parent, child, **attrs))

    # neither - unrelated
    if unrelated:
        for parent, child, label in neither:
            attrs = {'style': 'dashed',
                     'color': 'grey'}
            graph.add_edge(pydot.Edge(parent, child, **attrs))
    return graph
# pylint: enable=star-args
