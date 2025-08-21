# Reducing Class Imbalance Through Undersampling

""" This script creates a Train/Validate/Test split saving each split in a spereate .csv file to prevent data contamination. The 
    Training split is further processed using undersampling to create a balanced fraud to non-fraud ratio.  
"""

# Imports & Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Functions

