#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:44:42 2020

@author: susmitvengurlekar
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin,BaseEstimator

from common_transformers import DateHandler, TimeHandler


df = pd.read_csv("data/raw_data.csv")
pipe = Pipeline([("date_handler",DateHandler(["stop_date"],True,False)),("time_handler",TimeHandler(["stop_time"],True,False))])
df = pipe.fit_transform(df)
df.to_csv("data/date_time_handled_data.csv",index=False)


