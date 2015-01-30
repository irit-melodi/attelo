[![Build Status](https://secure.travis-ci.org/kowey/attelo.png)](http://travis-ci.org/kowey/attelo)
[![Documentation Status](https://readthedocs.org/projects/attelo/badge/?version=latest)](https://readthedocs.org/projects/attelo/?badge=latest)

## About

Attelo is a discourse parser. It predicts a discourse graph out of a set
of elementary discourse units. These predictions are informed by models
that allow the parser to assign probabilities to the links.

Attelo can be used as a library, or (more likely) as a standalone
program with the following subcommands:

* attelo learn: Build attachment/relation models from (CSV) feature files
* attelo decode: (the parser proper) Predict links given models (from
  `attelo learn`) and a set of features
* attelo evaluate: Cross-fold evaluation a pair of feature files

## Documentation

* [documentation root][docroot]
* [API][apidoc]

## Requirements

As a minimum, you will need a way to extract features from your
discourse corpus, and from any new inputs outside of your corpus. The
[educe library][educe] provides this functionality for a handful of
pre-existing corpora, and can also be used to build feature extractors
for your own corpora.

If you are using attelo in the context of discourse parsing experiments,
we highly recommend setting up some sort of project-specific
experimental harness to manage all of the variables you may
want throw into your experiment. See the [irit-rst-dt][irit-rst-dt]
experiment for an example of what such a harness would look like. (See
also the [Shake build system][shake]).

NB: The `attelo evaluate` command can be seen as a harness of
sorts but is bit limited at the moment. Our project harnessses use it
as one of its component, but throw in things like feature extraction,
and some basic looping around decoder/learner types. That said, it is
possible that `attelo evaluate` will grow some features of its own
and reduce the amount of infrastructure you need to build.

## Usage

Discourse parsing (rough sketch, see the `--help`):

1. extract features (DIY)
2. attelo learn
3. attelo decode

Running experiments (old way):

1. extract features (DIY)
2. attelo evaluate

### Finer grained evaluation

There is work in progress to break `attelo evaluate` up to avoid
repeated work. The idea would be to save intermediary models for
each fold and to re-use these models to test various decoders. The
new experiments would look a bit more like this:

1. extract features (DIY)
2. attelo enfold
3. for each fold: attelo learn --fold, attelo decode --fold
4. attelo report

See the requirements above. We recommend building an experimental
harness around attelo instead of trying to use it by hand. Even
a couple of shell scripts would be better than nothing.

## Authors

Attelo is developed within the IRIT Melodi team, based on work
originally by Stergos Afantenos, Pascal Denis, and Philippe Muller.
At the time of this writing (2015-02) it is maintained by Eric Kow.

## License (GPL-3)

Attelo is based on the Orange machine learning toolkit, which is
licensed under the GPL-3.

The Attelo components by themselves should be considered released
under the more liberal BSD3-like CeCILL-B license.  So long as
attelo depends on Orange, the combination should be considered as
GPL'ed however.

Aside from conformance with French law, the difference between the
CeCILL-B and other similarly liberal licenses is the strong citation
requirement mentioned in article 5.3.4 quoted below:

> Any Licensee who may distribute a Modified Software hereby expressly
> agrees to:
> 
>    1. indicate in the related documentation that it is based on the
>       Software licensed hereunder, and reproduce the intellectual
>       property notice for the Software,
> 
>    2. ensure that written indications of the Software intended use,
>       intellectual property notice and license hereunder are included in
>       easily accessible format from the Modified Software interface,
> 
>    3. mention, on a freely accessible website describing the Modified
>       Software, at least throughout the distribution term thereof, that
>       it is based on the Software licensed hereunder, and reproduce the
>       Software intellectual property notice,
> 
>    4. where it is distributed to a third party that may distribute a
>       Modified Software without having to make its source code
>       available, make its best efforts to ensure that said third party
>       agrees to comply with the obligations set forth in this Article .
> 
> If the Software, whether or not modified, is distributed with an
> External Module designed for use in connection with the Software, the
> Licensee shall submit said External Module to the foregoing obligations.

[docroot]: http://attelo.readthedocs.org/
[apidoc]: http://attelo.readthedocs.org/en/latest/api-doc/attelo/
[educe]: http://github.com/kowey/educe
[irit-rst-dt]: http://github.com/kowey/irit-rst-dt
[shake]: http://community.haskell.org/~ndm/shake/
