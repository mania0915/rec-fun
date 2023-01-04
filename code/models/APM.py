#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        :APM.py
@desc        :
@date        :2022/12/30 11:58:35
@author      :eason
@version     :1.0
'''


import pandas as pd
import numpy as np
import tqdm as tqdm
import warnings, random, math, os
from collections import namedtuple,OrderedDict

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_norlmal
from tensorflow.python.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinmaxScaler, StandardScaler, LabelEncoder

from utils import DenseFeat, SparseFeat, VarLenSparseFeat
import itertools