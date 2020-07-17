#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon 13 July 18:47:15 2020
@author: mcruces

################################
#    Miguel Cruces Fernández   #
#         -------------        #
#    mcsquared.fz@gmail.com    #
#   miguel.cruces@rai.usc.es   #
################################

"""


import os
from os.path import join as joinPath
import sys  # para debug: sys.exit()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

class CollectData:
    """
    Class that collects data from the *archive.txt* given
    """

    def __init__(self, rel_path: str,
                 year: int = 0,
                 dt_corr: float = 0,
                 date_fmt: str = 'sex',
                 separator: str = ', ',
                 titles: list = None,
                 time_column_idx: int = 0):

        rootdir = os.path.abspath("./")
        self.txtPath = joinPath(rootdir, rel_path)
        self.year = year
        self.dt_corr = dt_corr
        self.date_fmt = date_fmt
        self.separator = separator
        self.titles = titles
        self.time_column_idx = time_column_idx
        self.time_key = None