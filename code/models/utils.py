#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        :utils.py
@desc        :
@date        :2022/12/30 14:48:28
@author      :eason
@version     :1.0
'''

from collections import namedtuple

SparseFeat = namedtuple('SparseFeat',
                        ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple(
    'VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])


