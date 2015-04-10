#!/usr/bin/env python
# pylint: disable=invalid-name
# pylint: enable=invalid-name

'''
List the local evaluation configurations on stdout
'''

from __future__ import print_function
from irit_rst_dt.local import print_evaluations
from irit_rst_dt.util import (sanity_check_config,
                              test_evaluation)

if __name__ == '__main__':
    print_evaluations()
    print('TEST eval:', test_evaluation())
    sanity_check_config()
