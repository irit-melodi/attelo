[![Build Status](https://secure.travis-ci.org/irit-melodi/attelo.png)](http://travis-ci.org/irit-melodi/attelo)
[![Coverage Status](https://coveralls.io/repos/irit-melodi/attelo/badge.svg?branch=master)](https://coveralls.io/r/irit-melodi/attelo?branch=master)
[![Documentation Status](https://readthedocs.org/projects/attelo/badge/?version=latest)](https://readthedocs.org/projects/attelo/?badge=latest)

## About

Attelo is a discourse parsing library. The parsers predict a a discourse
graph out of a set of elementary discourse units. These predictions are
informed by models that allow the parser to assign probabilities to the
links.

## Documentation

* [documentation root][docroot]
* [API][apidoc]

## Requirements

As a minimum, you will need a way to extract features from your
discourse corpus, and from any new inputs outside of your corpus. The
[educe library][educe] provides this functionality for a handful of
pre-existing corpora, and can also be used to build feature extractors
for your own corpora.

It would probably be a good idea to build some sort of test harness for
your discourse experiments (have a look at the
[quickstart][quickstart]).  For more detailed experiments,
you might use the [irit-rst-dt][irit-rst-dt] experiment as a starting
point, or alternatively, build something off the
[Shake build system][shake].

NB: The `attelo evaluate` command can be seen as a harness of
sorts but is bit limited at the moment. Our project harnessses use it
as one of its component, but throw in things like feature extraction,
and some basic looping around decoder/learner types. That said, it is
possible that `attelo evaluate` will grow some features of its own
and reduce the amount of infrastructure you need to build.

## Usage

Discourse parsing (rough sketch, see the `--help`):

1. extract features (DIY)
2. use attelo to enfold, learn, decode, and score
   (see [quickstart][quickstart])

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

## License (Cecill-B)

Attelo is released under a liberal BSD3-like CeCILL-B license.
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
[educe]: http://github.com/irit-melodi/educe
[irit-rst-dt]: http://github.com/irit-melodi/irit-rst-dt
[quikstart]: http://attelo.readthedocs.org/en/latest/quickstart/
[shake]: http://community.haskell.org/~ndm/shake/
