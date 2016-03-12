# GENERATED FILE, do not edit by hand
# Source: src/PyTorch.jinja2.pyx

from __future__ import print_function, division
import numbers
import cython
cimport cython

import numpy as np

cimport cpython.array
import array

from math import log10, floor

from lua cimport *
cimport PyTorch

# from http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)


# GENERATED FILE, do not edit by hand
# Source: src/PyTorch.jinja2.pxd

cdef extern from "THRandom.h":
    cdef struct THGenerator

#cdef struct lua_State
#cdef struct THGenerator

cdef extern from "THTensor.h":
    cdef struct THLongTensor






