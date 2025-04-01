import warnings
warnings.filterwarnings('ignore')  # Check warnings before production
import os
import re
import base64
import json
import yaml
from itertools import product
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import awswrangler as wr
import boto3

