# Transforming and Normalizing the BAF Dataset

""" This Python script converts the four categorical variables into ordinal values based on insights from the previous univariate 
    analysis. Additionally, numeric variables are normalized to ensure a consistent scaling of values across variables.
"""

# Imports & Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Functions



# Ordinal transformations
payment_dict = {"AA": 1, "AB": 4, "AC": 5, "AD": 3, "AE": 2}

housing_dict = {"BA": 7, "BB": 4, "BC": 5, "BD": 6, "BE": 1, "BF": 3, "BG": 2}

employment_dict = {"CA": 5, "CB": 4, "CC": 7, "CD": 3, "CE": 1, "CF": 2, "CG": 6}

device_dict = {"linux": 1, "macintosh": 4, "other": 2, "x11": 3, "windows": 5}

