import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import openpyxl
import argparse
import os

def check_dependencies():
    """
    检查所需的依赖是否已加载。
    """
    try:
        print("Dependencies loaded successfully!")
    except ImportError as e:
        print(f"Error loading dependencies: {e}")

