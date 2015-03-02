"graph visualisation"

from __future__ import print_function
from collections import (defaultdict, namedtuple)
from os import path as fp
import itertools as itr
import codecs
import signal
import subprocess
import sys

import pydot

from .edu import FAKE_ROOT_ID
from .harness.util import makedirs
from .table import UNRELATED

# pylint: disable=too-few-public-methods

DEFAULT_TIMEOUT = 30


class Alarm(Exception):
    "Exception to raise on signal timeout"
    pass


# pylint: disable=unused-argument
def alarm_handler(_, frame):
    "Raise Alarm on signal"
    raise Alarm
# pylint: enable=unused-argument


class GraphSettings(namedtuple('GraphSettings',
                               ['hide',
                                'select',
                                'unrelated',
                                'timeout',
                                'quiet'])):
    '''
    :param hide: 'intra' to hide links between EDUs in the
                 same subgrouping; 'inter' to hide links
                 across subgroupings; None to show all links
    :type hide: string or None

    :param select: EDU groupings to graph (if None,
                   all groupings will be graphed unless)
    :type select: [string] or None

    :param unrelated: show unrelated links
    :type unrelated: bool

    :param timeout: number of seconds to allow graphviz
                    to run before it times out
    :type timeout: int

    :param quiet: suppress informational messages
    :type quiet: bool
    '''
    pass


def select_links(edus, links, settings):
    """
    Given a set of edus and of edu id pairs, return only the pairs
    whose ids appear in the edu list

    :param intra: if True, in addition to the constraints above,
                  only return links that are in the same subgrouping
    :param inter: if True, only return links between subgroupings
    """
    subgroupings = {edu.id: edu.subgrouping for edu in edus}
    edu_ids = subgroupings.keys()

    if settings.hide == 'inter':
        slinks = [(subgroupings.get(e1, FAKE_ROOT_ID),
                   subgroupings.get(e2, FAKE_ROOT_ID), l)
                  for e1, e2, l in links]
        return [(s1, s2, l) for (s1, s2, l) in slinks if s1 != s2]
    elif settings.hide == 'intra':
        return [(e1, e2, l) for e1, e2, l in links
                if (e1 in edu_ids or e2 in edu_ids)
                and subgroupings.get(e1) == subgroupings.get(e2)]
    else:
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


def mk_diff_graph(title, edus, src_links, tgt_links, settings):
    """
    Convert attelo predictions to a graphviz graph diplaying
    differences between two predictions

    Predictions here consist of an EDU followed by a list of
    (parent name, relation label) tuples

    :param tgt_links: if present, we generate a graph that
                      represents a difference between the
                      links and tgt_links (by highlighting
                      links that only occur in one or the
                      other)
    """
    graph = _build_core_graph(title, edus,
                              inter=settings.hide == 'intra')
    both, src_only, tgt_only, neither = _diff_links(src_links,
                                                    tgt_links or [])
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
    if settings.unrelated:
        for parent, child, label in neither:
            attrs = {'style': 'dashed',
                     'color': 'grey'}
            graph.add_edge(pydot.Edge(parent, child, **attrs))
    return graph
# pylint: enable=star-args


def mk_single_graph(title, edus, links, settings):
    """
    Convert single set of attelo predictions to a graphviz
    graph
    """
    return mk_diff_graph(title, edus, links, None, settings)


def diff_all(edus,
             src_predictions,
             tgt_predictions,
             settings,
             output_dir):
    """
    Generate graphs for all the given predictions.
    Each grouping will have its own graph, saved in the
    output directory
    """
    for group, subedus_ in itr.groupby(edus, lambda x: x.grouping):
        if settings.select is not None and group not in settings.select:
            continue
        subedus = list(subedus_)
        src_sublinks = select_links(subedus, src_predictions, settings)
        if not src_sublinks:  # not in fold
            continue
        # skip any groups that are not in diff target (envisioned
        # use case, diffing gold against an output)
        if tgt_predictions is None:
            tgt_sublinks = None
        else:
            tgt_sublinks = select_links(subedus, tgt_predictions, settings)
            if not tgt_sublinks:
                continue
        graph = mk_diff_graph(group, subedus,
                              src_sublinks,
                              tgt_sublinks,
                              settings=settings)
        ofilename = fp.join(output_dir, group)
        write_dot_graph(ofilename, graph,
                        quiet=settings.quiet,
                        timeout=settings.timeout)


def graph_all(edus,
              predictions,
              settings,
              output_dir):
    """
    Generate graphs for all the given predictions.
    Each grouping will have its own graph, saved in the
    output directory
    """
    diff_all(edus, predictions, None, settings, output_dir)
