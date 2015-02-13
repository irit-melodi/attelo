"show properties about models"

from __future__ import print_function
from itertools import chain
import codecs
import itertools
import six

from tabulate import tabulate
import numpy

from attelo.args import (add_model_read_args)
from attelo.io import (load_labels, load_model, load_vocab)
from attelo.table import (UNRELATED)

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


def condense_cell(old, new):
    """
    Maximise readability of the new cell given that it's sitting
    below the old one in a 2D table
    """
    if isinstance(new, six.string_types):
        is_eqish = lambda (x, y): x == y and '=' not in [x, y]
        zipped = list(itertools.izip_longest(old, new))
        prefix = itertools.takewhile(is_eqish, zipped)
        suffix = itertools.dropwhile(is_eqish, zipped)
        return ''.join(['.' for _ in prefix] +
                       [n if n is not None else '' for _, n in suffix])
    else:
        return '{:.2f}'.format(new)


def condense_table(rows):
    """
    Make a table more readable by replacing identical columns in
    subsequent rows by "
    """
    if not rows:
        return rows
    results = []
    current_row = ['' for _ in rows[0]]
    for row in rows:
        new_row = [row[0]]
        new_row.extend(condense_cell(old, new)
                       for old, new in zip(current_row[1:], row[1:]))
        results.append(new_row)
        current_row = row
    return results


def sort_table(rows):
    """
    Return rows in the following order

    * UNRELATED always comes first
    * otherwise, sort by the names of top N features

    The hope is that this would visually group together the same
    features so you can see a natural separation
    """
    label_value = {'UNRELATED': -2}

    def ordering_key(row):
        "tweaked version of list of sorting"
        label = label_value.get(row[0], 0)
        rest = row[1::2]
        return (label, rest)

    return sorted(rows, key=ordering_key)


def _best_feature_indices(vocab, model, class_index, top_n):
    """
    Return a list of strings representing the best features in
    a model for a given class index
    """
    weights = model.coef_[class_index]   # higher is better?
    # pylint: disable=no-member
    best_idxes = numpy.argsort(weights)[-top_n:][::-1]
    best_weights = numpy.take(weights, best_idxes)
    # pylint: enable=no-member
    res = chain.from_iterable([vocab[j], w]
                              for j, w in zip(best_idxes, best_weights))
    return list(res)


def main_for_harness(args):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    attach_model = load_model(args.attachment_model)
    relate_model = load_model(args.relation_model)
    labels = load_labels(args.features)
    vocab = load_vocab(args.vocab)

    best_idxes = lambda m, i: _best_feature_indices(vocab, m, i, args.top)

    rows = []
    rows.append([UNRELATED] + best_idxes(attach_model, 0))
    for i, class_ in enumerate(relate_model.classes_):
        label = labels[int(class_) - 1]
        rows.append([label] + best_idxes(relate_model, i))

    res = tabulate(condense_table(sort_table(rows)))
    if args.output is None:
        print(res)
    else:
        with codecs.open(args.output, 'wb', 'utf-8') as fout:
            print(res, file=fout)


def main(args):
    "subcommand main (invoked from outer script)"

    main_for_harness(args)
