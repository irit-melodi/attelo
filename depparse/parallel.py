# Copyright (c) 2010 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''Utilities to provide parallel processing for Sentence tasks.'''

import time
import logging
import multiprocessing


def iterqueue(q, name='items', log_interval=0):
    '''Iterate over all items immediately available in a Queue.'''
    start = time.time()

    def elapsed():
        count = time.time() - start
        unit = 's'
        if count < 10:
            count *= 1000
            unit = 'ms'
        return count, unit

    count = 0
    while True:
        try:
            yield q.get(timeout=0.3)
            count += 1
        except:
            logging.warning('processed %d %s in %d%s', count, name, *elapsed())
            break
        if log_interval and not count % log_interval:
            logging.info('processed %d %s in %d%s', count, name, *elapsed())


def launch(concurrency, sents, target, parser, mkargs=None, **kwargs):
    '''Launch a set of jobs concurrently to process some Sentences.

    concurrency: The number of concurrent workers to launch.
    sents: A sequence of Sentences to process.
    target: The target for the concurrent workers to execute.
    parser: A Parser to run over the Sentences.
    mkargs: If provided, this should be a callable that takes a worker index and
      returns a (possibly empty) tuple of args to pass to that worker.
    kwargs: A dictionary of extra args to pass to the workers.

    Return a Queue from which results may be obtained with get().
    '''
    sent_q = multiprocessing.Queue()
    multiprocessing.Process(target=_enqueue, args=(sent_q, sents)).start()
    result_q = multiprocessing.Queue()
    for i in xrange(concurrency):
        extra_args = ()
        if callable(mkargs):
            extra_args = mkargs(i)
        args = (sent_q, result_q, parser) + extra_args
        multiprocessing.Process(target=target, args=args, kwargs=kwargs).start()
    return result_q


def _enqueue(sent_q, sents):
    '''Put a bunch of Sentences on a queue.'''
    for sent in sents:
        sent_q.put(sent)
    logging.warning('added %d sentences to processing queue', len(sents))


def _train(sent_q, result_q, parser, action_classifier, label_classifier, **kwargs):
    '''Run a training loop in one process until there are no more sentences.
    '''
    for sent in iterqueue(sent_q, 'training sentences', 50):
        logging.debug('training on [%s]', sent)
        parser.learn(sent, action_classifier, label_classifier, **kwargs)
    result_q.put((action_classifier, label_classifier))


def _test(sent_q, result_q, parser, **kwargs):
    '''Run a testing loop in one process until there are no more sentences.
    '''
    for sent in iterqueue(sent_q, 'testing sentences', 50):
        logging.debug('testing on [%s]', sent)
        result_q.put((parser.parse(sent, **kwargs), sent))
