"learn and save models"

from __future__ import print_function
from os import path as fp

from joblib import (Parallel, delayed)

from ..args import\
    (add_common_args,
     add_learner_args, validate_learner_args,
     add_fold_choice_args, validate_fold_choice_args,
     args_to_decoder, args_to_learners)
from ..harness.util import (makedirs)
from ..io import (Torpor, save_model, load_fold_dict)
from ..learning import (learn_attach, learn_relate)
from ..table import (for_intra)
from .util import load_args_data_pack


_DEFAULT_MODEL_ATTACH = "attach.model"
_DEFAULT_MODEL_RELATION = "relations.model"

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _load_and_select_data(args):
    """
    read data and filter on fold if relevant
    """
    if args.fold is None:
        dpack = load_args_data_pack(args)
    else:
        # load data pack *AFTER* fold dict (fail faster)
        fold_dict = load_fold_dict(args.fold_file)
        dpack = load_args_data_pack(args)
        dpack = dpack.training(fold_dict, args.fold)

    if args.intrasentential:
        dpack = for_intra(dpack)

    return dpack


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def config_argparser(psr):
    "add subcommand arguments to subparser"

    add_common_args(psr)
    add_learner_args(psr)
    add_fold_choice_args(psr)
    psr.add_argument("--attachment-model", "-A", metavar="FILE",
                     default=_DEFAULT_MODEL_ATTACH,
                     help="save attachment model here "
                     "(default: %s)" % _DEFAULT_MODEL_ATTACH)
    psr.add_argument("--relation-model", "-R", metavar="FILE",
                     default=_DEFAULT_MODEL_RELATION,
                     help="save relation model here "
                     "(default: %s)" % _DEFAULT_MODEL_RELATION)
    psr.set_defaults(func=main)


def main_for_harness(args, dpack):
    """
    main function core that you can hook into if writing your own
    harness

    You have to supply (and filter) the data yourself
    (see `select_data`)
    """
    Parallel(n_jobs=-1)(delayed_main_for_harness(args, dpack))


def _announce(task, model_path, quiet=True):
    'return a Torpor context manager to bracket a learning job'
    msg = ("training {task} model {path}"
           "").format(task=task,
                      path=fp.basename(model_path))
    return Torpor(msg, sameline=False, quiet=quiet)


def _learn_and_save(dpack, learn_fn, model_path):
    'learn and write the model'
    model = learn_fn(dpack)
    makedirs(fp.dirname(model_path))
    save_model(model_path, model)


def learn_and_save_attach(args, learners, dpack):
    'learn and write the attachment model'
    path = args.attachment_model
    learn_fn = lambda x: learn_attach(learners, x)
    with _announce('attachment', path, args.quiet):
        _learn_and_save(dpack, learn_fn, path)


def learn_and_save_relate(args, learners, dpack):
    'learn and write the relation model'
    path = args.relation_model
    learn_fn = lambda x: learn_relate(learners, x)
    with _announce('relation', path, args.quiet):
        _learn_and_save(dpack, learn_fn, path)


def delayed_main_for_harness(args, dpack):
    """
    Advanced variant of the main function which returns a list
    of model-learning futures that will have to be executed later.
    The idea is that you should be able to build up a large list
    of parallel model-learning tasks to execute simultaneously
    and just go for it
    """
    decoder = args_to_decoder(args)
    learners = args_to_learners(decoder, args)
    return [delayed(learn_and_save_attach)(args, learners, dpack),
            delayed(learn_and_save_relate)(args, learners, dpack)]


@validate_fold_choice_args
@validate_learner_args
def main(args):
    "subcommand main (invoked from outer script)"

    dpack = _load_and_select_data(args)
    main_for_harness(args, dpack)
